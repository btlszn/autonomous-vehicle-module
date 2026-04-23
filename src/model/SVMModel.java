package model;

import data.DataPoint;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Eğitilmiş SVM modelini temsil eder.
 * Immutable model: eğitim sonrası değiştirilemez.
 */
public final class SVMModel {

    private final double[] weights;          // Ağırlık vektörü (w)
    private final double bias;               // Bias terimi (b)
    private final double margin;             // Toplam marjin genişliği
    private final List<DataPoint> supportVectors; // Destek vektörleri

    public SVMModel(double[] weights, double bias, double margin,
                    List<DataPoint> supportVectors) {
        if (weights == null || weights.length == 0) {
            throw new IllegalArgumentException("Weights boş olamaz.");
        }
        this.weights = Arrays.copyOf(weights, weights.length);
        this.bias = bias;
        this.margin = margin;
        this.supportVectors = Collections.unmodifiableList(supportVectors);
    }

    /**
     * Bir noktanın sınıfını tahmin eder.
     *
     * @param features Özellik vektörü
     * @return +1 veya -1
     */
    public int predict(double[] features) {
        double score = decisionScore(features);
        return score >= 0 ? 1 : -1;
    }

    /**
     * Ham karar fonksiyonu skoru: w·x + b
     * Pozitif -> Sınıf +1, Negatif -> Sınıf -1
     */
    public double decisionScore(double[] features) {
        if (features.length != weights.length) {
            throw new IllegalArgumentException("Boyut uyumsuzluğu.");
        }
        double score = bias;
        for (int i = 0; i < weights.length; i++) {
            score += weights[i] * features[i];
        }
        return score;
    }

    /**
     * ||w|| - ağırlık vektörünün normu
     */
    public double getWeightNorm() {
        double sumSq = 0.0;
        for (double w : weights) {
            sumSq += w * w;
        }
        return Math.sqrt(sumSq);
    }

    /**
     * Bir noktanın karar sınırına olan geometrik mesafesi
     */
    public double geometricDistance(double[] features) {
        return Math.abs(decisionScore(features)) / getWeightNorm();
    }

    // --- Getter'lar ---

    public double[] getWeights() {
        return Arrays.copyOf(weights, weights.length);
    }

    public double getBias() {
        return bias;
    }

    public double getMargin() {
        return margin;
    }

    public List<DataPoint> getSupportVectors() {
        return supportVectors;
    }

    @Override
    public String toString() {
        return String.format("SVMModel{weights=%s, bias=%.4f, margin=%.4f, supportVectors=%d}",
                Arrays.toString(weights), bias, margin, supportVectors.size());
    }
}
