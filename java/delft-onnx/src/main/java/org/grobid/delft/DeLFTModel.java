package org.grobid.delft;

import ai.onnxruntime.OrtException;
import org.grobid.core.layout.LayoutToken;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Closeable;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Main entry point for DeLFT ONNX model inference.
 * 
 * Loads an exported model and provides text annotation functionality.
 * Supports both BidLSTM_CRF (no features) and BidLSTM_CRF_FEATURES models.
 */
public class DeLFTModel implements Closeable {

    private static final Logger LOGGER = LoggerFactory.getLogger(DeLFTModel.class);

    private final OnnxModelRunner modelRunner;
    private final CRFDecoder crfDecoder;
    private final Preprocessor preprocessor;
    private final WordEmbeddings embeddings;
    private final int maxSeqLength;

    /**
     * Load a DeLFT model from exported directory.
     * 
     * @param modelDir       Directory containing encoder.onnx, crf_params.json,
     *                       vocab.json
     * @param embeddingsPath Path to LMDB embeddings database
     * @param embeddingSize  Dimension of word embeddings
     * @param maxSeqLength   Maximum sequence length
     */
    public DeLFTModel(Path modelDir, Path embeddingsPath, int embeddingSize, int maxSeqLength)
            throws IOException, OrtException {

        this.maxSeqLength = maxSeqLength;

        // Load components
        this.modelRunner = new OnnxModelRunner(modelDir.resolve("encoder.onnx"));
        this.crfDecoder = CRFDecoder.fromJson(modelDir.resolve("crf_params.json"));
        this.preprocessor = Preprocessor.fromJson(modelDir.resolve("vocab.json"));
        this.embeddings = new WordEmbeddings(embeddingsPath, embeddingSize);

        LOGGER.info("DeLFT model loaded from {}", modelDir);
        LOGGER.info("Model has features: {}", preprocessor.hasFeatures());
    }

    /**
     * Annotate text with sequence labels (no features).
     * 
     * @param text Input text
     * @return Annotation result
     */
    public AnnotationResult annotate(String text) throws OrtException {
        // Tokenize
        List<LayoutToken> tokens = preprocessor.tokenize(text);
        String[] words = new String[tokens.size()];
        for (int i = 0; i < tokens.size(); i++) {
            words[i] = tokens.get(i).getText();
        }

        return annotateTokens(words, null);
    }

    /**
     * Annotate tokens with features.
     * 
     * For BidLSTM_CRF_FEATURES models, features must be provided.
     * Each row in features corresponds to a token, with one value per feature
     * column.
     * 
     * @param tokens   Array of token strings
     * @param features Feature values per token [numTokens][numFeatures], can be
     *                 null for non-feature models
     * @return Annotation result
     */
    public AnnotationResult annotateTokens(String[] tokens, String[][] features) throws OrtException {
        int numTokens = Math.min(tokens.length, maxSeqLength);

        if (numTokens == 0) {
            return new AnnotationResult(null, new String[0], new String[0]);
        }

        // Truncate to max sequence length
        String[] words = new String[numTokens];
        System.arraycopy(tokens, 0, words, 0, numTokens);

        // Get embeddings [seq_len][embed_size]
        float[][] wordEmbs = embeddings.getEmbeddings(words);

        // Pad to maxSeqLength
        float[][] paddedEmbs = new float[maxSeqLength][embeddings.getEmbeddingSize()];
        for (int i = 0; i < numTokens; i++) {
            paddedEmbs[i] = wordEmbs[i];
        }

        // Get char indices [seq_len][max_char]
        List<LayoutToken> layoutTokens = new ArrayList<>();
        for (String word : words) {
            LayoutToken lt = new LayoutToken();
            lt.setText(word);
            layoutTokens.add(lt);
        }
        long[][] charIndices = preprocessor.tokensToCharIndices(layoutTokens, maxSeqLength);

        // Create batch of 1
        float[][][] batchEmbs = new float[][][] { paddedEmbs };
        long[][][] batchChars = new long[][][] { charIndices };

        // Handle features
        long[][][] batchFeatures = null;
        if (preprocessor.hasFeatures() && features != null) {
            long[][] featureIndices = preprocessor.tokensToFeatureIndices(features, maxSeqLength);
            batchFeatures = new long[][][] { featureIndices };
        }

        // Run model
        float[][][] emissions = modelRunner.runInference(batchEmbs, batchChars, batchFeatures);

        // CRF decode
        boolean[] mask = preprocessor.createMask(numTokens, maxSeqLength);
        int[] tagIndices = crfDecoder.decode(emissions[0], mask);

        // Convert to tag names (only for actual tokens)
        String[] tags = new String[numTokens];
        for (int i = 0; i < numTokens; i++) {
            tags[i] = preprocessor.getTagIndex().getOrDefault(tagIndices[i], "O");
        }

        return new AnnotationResult(String.join(" ", words), words, tags);
    }

    /**
     * Check if this model requires features.
     */
    public boolean hasFeatures() {
        return preprocessor.hasFeatures();
    }

    /**
     * Get the number of features expected per token (0 if no features).
     */
    public int getNumFeatures() {
        return preprocessor.getNumFeatures();
    }

    /**
     * Annotate multiple texts in batch.
     */
    public List<AnnotationResult> annotateBatch(List<String> texts) throws OrtException {
        List<AnnotationResult> results = new ArrayList<>();
        for (String text : texts) {
            results.add(annotate(text));
        }
        return results;
    }

    @Override
    public void close() {
        if (modelRunner != null)
            modelRunner.close();
        if (embeddings != null)
            embeddings.close();
    }

    /**
     * Annotation result containing tokens and labels.
     */
    public static class AnnotationResult {
        private final String text;
        private final String[] tokens;
        private final String[] labels;

        public AnnotationResult(String text, String[] tokens, String[] labels) {
            this.text = text;
            this.tokens = tokens;
            this.labels = labels;
        }

        public String getText() {
            return text;
        }

        public String[] getTokens() {
            return tokens;
        }

        public String[] getLabels() {
            return labels;
        }

        public String toJson() {
            StringBuilder sb = new StringBuilder();
            sb.append("{\n");
            sb.append("  \"text\": \"").append(escapeJson(text)).append("\",\n");
            sb.append("  \"tokens\": [");
            for (int i = 0; i < tokens.length; i++) {
                if (i > 0)
                    sb.append(", ");
                sb.append("\"").append(escapeJson(tokens[i])).append("\"");
            }
            sb.append("],\n");
            sb.append("  \"labels\": [");
            for (int i = 0; i < labels.length; i++) {
                if (i > 0)
                    sb.append(", ");
                sb.append("\"").append(labels[i]).append("\"");
            }
            sb.append("]\n}");
            return sb.toString();
        }

        private String escapeJson(String s) {
            return s.replace("\\", "\\\\")
                    .replace("\"", "\\\"")
                    .replace("\n", "\\n")
                    .replace("\r", "\\r")
                    .replace("\t", "\\t");
        }
    }
}
