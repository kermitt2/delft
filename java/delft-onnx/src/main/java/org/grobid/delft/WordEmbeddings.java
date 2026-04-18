package org.grobid.delft;

import org.lmdbjava.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Word embeddings lookup using LMDB database.
 * 
 * Reads embeddings from LMDB where values are raw float32 arrays
 * (little-endian).
 * Use convert_lmdb_embeddings.py to convert from pickled numpy format.
 */
public class WordEmbeddings implements Closeable {

    private static final Logger LOGGER = LoggerFactory.getLogger(WordEmbeddings.class);

    private final Env<ByteBuffer> env;
    private final Dbi<ByteBuffer> dbi;
    private final int embeddingSize;
    private final float[] zeroVector;

    /**
     * Open LMDB database for word embeddings.
     * 
     * @param dbPath        Path to the LMDB database directory
     * @param embeddingSize Dimension of the embeddings
     * @throws IOException if the database path does not exist
     */
    public WordEmbeddings(Path dbPath, int embeddingSize) throws IOException {
        this.embeddingSize = embeddingSize;
        this.zeroVector = new float[embeddingSize];

        // Check if path exists before trying to open
        if (!Files.exists(dbPath)) {
            throw new IOException("Embeddings database not found: " + dbPath.toAbsolutePath() +
                    "\nPlease provide a valid path to an LMDB embeddings database.");
        }
        if (!Files.isDirectory(dbPath)) {
            throw new IOException("Embeddings path is not a directory: " + dbPath.toAbsolutePath() +
                    "\nLMDB databases are directories containing 'data.mdb' and 'lock.mdb' files.");
        }

        // Open LMDB environment
        this.env = Env.create()
                .setMapSize(10_000_000_000L) // 10GB max
                .setMaxDbs(1)
                .open(dbPath.toFile());

        // Open the default database
        this.dbi = env.openDbi((String) null, DbiFlags.MDB_CREATE);

        LOGGER.info("Opened LMDB database at {}", dbPath);
    }

    /**
     * Look up embedding for a word.
     * 
     * @param word The word to look up
     * @return Embedding vector, or zero vector if not found
     */
    public float[] getEmbedding(String word) {
        // Normalize digits to "0" like Python's _normalize_num
        String normalizedWord = normalizeNum(word);

        byte[] keyBytes = normalizedWord.getBytes(StandardCharsets.UTF_8);
        ByteBuffer keyBuffer = ByteBuffer.allocateDirect(keyBytes.length);
        keyBuffer.put(keyBytes).flip();

        try (Txn<ByteBuffer> txn = env.txnRead()) {
            ByteBuffer valueBuffer = dbi.get(txn, keyBuffer);

            if (valueBuffer == null) {
                // Word not found, return zero vector
                return zeroVector.clone();
            }

            // Parse float array from raw bytes (little-endian float32)
            valueBuffer.order(ByteOrder.LITTLE_ENDIAN);
            float[] embedding = new float[embeddingSize];
            for (int i = 0; i < embeddingSize; i++) {
                embedding[i] = valueBuffer.getFloat();
            }
            return embedding;
        }
    }

    /**
     * Normalize digits in a word to "0" (matches Python's _normalize_num).
     * This is needed because the model was trained with this normalization.
     * 
     * @param word Input word
     * @return Word with all digits replaced by "0"
     */
    private String normalizeNum(String word) {
        StringBuilder sb = new StringBuilder();
        for (char c : word.toCharArray()) {
            if (Character.isDigit(c)) {
                sb.append('0');
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    /**
     * Look up embeddings for a sequence of words.
     * 
     * @param words Array of words
     * @return 2D array [seq_len][embedding_size]
     */
    public float[][] getEmbeddings(String[] words) {
        float[][] embeddings = new float[words.length][embeddingSize];
        for (int i = 0; i < words.length; i++) {
            embeddings[i] = getEmbedding(words[i]);
        }
        return embeddings;
    }

    /**
     * Check if a word exists in the database.
     */
    public boolean contains(String word) {
        byte[] keyBytes = word.getBytes(StandardCharsets.UTF_8);
        ByteBuffer keyBuffer = ByteBuffer.allocateDirect(keyBytes.length);
        keyBuffer.put(keyBytes).flip();

        try (Txn<ByteBuffer> txn = env.txnRead()) {
            return dbi.get(txn, keyBuffer) != null;
        }
    }

    public int getEmbeddingSize() {
        return embeddingSize;
    }

    @Override
    public void close() {
        if (dbi != null) {
            dbi.close();
        }
        if (env != null) {
            env.close();
        }
    }
}
