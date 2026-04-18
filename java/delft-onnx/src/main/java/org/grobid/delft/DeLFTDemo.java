package org.grobid.delft;

import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Command-line demonstrator for DeLFT ONNX inference.
 * 
 * Usage:
 * java -jar delft-onnx.jar --model <model_dir> --embeddings <lmdb_path>
 * --embedding-size 300 --input "text to annotate"
 * 
 * For feature models, use --test-features to run with sample features.
 */
public class DeLFTDemo {

    public static void main(String[] args) {
        String modelDir = null;
        String embeddingsPath = null;
        int embeddingSize = 300;
        int maxSeqLength = 512;
        String inputText = null;

        // Parse arguments
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--model":
                    modelDir = args[++i];
                    break;
                case "--embeddings":
                    embeddingsPath = args[++i];
                    break;
                case "--embedding-size":
                    embeddingSize = Integer.parseInt(args[++i]);
                    break;
                case "--max-seq-length":
                    maxSeqLength = Integer.parseInt(args[++i]);
                    break;
                case "--input":
                    inputText = args[++i];
                    break;
                case "--help":
                    printUsage();
                    return;
            }
        }

        // Validate
        if (modelDir == null || embeddingsPath == null || inputText == null) {
            System.err.println("Error: Missing required arguments");
            printUsage();
            System.exit(1);
        }

        try {
            // Load model
            System.err.println("Loading model from: " + modelDir);
            DeLFTModel model = new DeLFTModel(
                    Paths.get(modelDir),
                    Paths.get(embeddingsPath),
                    embeddingSize,
                    maxSeqLength);

            // Print model info
            System.err.println("Model has features: " + model.hasFeatures());
            if (model.hasFeatures()) {
                System.err.println("Number of features: " + model.getNumFeatures());
            }

            // Run inference
            System.err.println("Processing input: " + inputText);
            DeLFTModel.AnnotationResult result;

            if (model.hasFeatures()) {
                // Feature model: tokenize and generate sample features
                String[] tokens = inputText.split("\\s+");
                String[][] features = createSampleDateFeatures(tokens);
                System.err.println("Generated sample features for " + tokens.length + " tokens");
                result = model.annotateTokens(tokens, features);
            } else {
                result = model.annotate(inputText);
            }

            // Print result as JSON
            System.out.println(result.toJson());

            // Clean up
            model.close();

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * Create sample features for date model testing.
     * 
     * Features for date model include:
     * - Line position (LINESTART, LINEIN, LINEEND)
     * - Capitalization (ALLCAP, INITCAP, NOCAPS)
     * - Digit pattern (ALLDIGIT, CONTAINSDIGITS, NODIGIT)
     * - Boolean flags (0/1)
     * - Punctuation (COMMA, DOT, HYPHEN, NOPUNCT, etc.)
     */
    private static String[][] createSampleDateFeatures(String[] tokens) {
        int numFeatures = 7; // Date model has 7 features
        String[][] features = new String[tokens.length][numFeatures];

        for (int i = 0; i < tokens.length; i++) {
            String token = tokens[i];

            // Feature 0: Line position
            if (i == 0) {
                features[i][0] = "LINESTART";
            } else if (i == tokens.length - 1) {
                features[i][0] = "LINEEND";
            } else {
                features[i][0] = "LINEIN";
            }

            // Feature 1: Capitalization
            if (token.equals(token.toUpperCase()) && token.matches(".*[A-Z].*")) {
                features[i][1] = "ALLCAP";
            } else if (Character.isUpperCase(token.charAt(0))) {
                features[i][1] = "INITCAP";
            } else {
                features[i][1] = "NOCAPS";
            }

            // Feature 2: Digit pattern
            if (token.matches("\\d+")) {
                features[i][2] = "ALLDIGIT";
            } else if (token.matches(".*\\d.*")) {
                features[i][2] = "CONTAINSDIGITS";
            } else {
                features[i][2] = "NODIGIT";
            }

            // Feature 3-5: Boolean flags (default to 0)
            features[i][3] = "0";
            features[i][4] = "0";
            features[i][5] = "0";

            // Feature 6: Punctuation
            if (token.equals(",")) {
                features[i][6] = "COMMA";
            } else if (token.equals(".")) {
                features[i][6] = "DOT";
            } else if (token.equals("-")) {
                features[i][6] = "HYPHEN";
            } else if (token.matches("[()\\[\\]]")) {
                features[i][6] = token.matches("[\\(\\[]") ? "OPENBRACKET" : "ENDBRACKET";
            } else if (token.matches("\\p{Punct}")) {
                features[i][6] = "PUNCT";
            } else {
                features[i][6] = "NOPUNCT";
            }
        }

        return features;
    }

    private static void printUsage() {
        System.out.println("DeLFT ONNX Inference Demo");
        System.out.println();
        System.out.println("Usage:");
        System.out.println("  java -jar delft-onnx.jar [options]");
        System.out.println();
        System.out.println("Required Options:");
        System.out.println("  --model <path>          Path to exported model directory");
        System.out.println("  --embeddings <path>     Path to LMDB embeddings database");
        System.out.println("  --input <text>          Text to annotate");
        System.out.println();
        System.out.println("Optional:");
        System.out.println("  --embedding-size <n>    Embedding dimension (default: 300)");
        System.out.println("  --max-seq-length <n>    Max sequence length (default: 512)");
        System.out.println("  --help                  Show this help");
        System.out.println();
        System.out.println("Example:");
        System.out.println("  java -jar delft-onnx.jar \\");
        System.out.println("    --model ./exported_models/date-features \\");
        System.out.println("    --embeddings ./data/db/glove-840B \\");
        System.out.println("    --input \"December 25, 2024\"");
        System.out.println();
        System.out.println("Note: For feature models (BidLSTM_CRF_FEATURES), sample features are");
        System.out.println("      auto-generated for demo purposes.");
    }
}
