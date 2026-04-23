package data;

import java.util.Arrays;
import java.util.Objects;

/**
 * Bir veri noktasını temsil eder.
 * Koordinatlar ve sınıf etiketi içerir.
 */
public final class DataPoint {

    private final double[] features;  // Özellik vektörü (koordinatlar)
    private final int label;           // Sınıf etiketi: +1 veya -1

    /**
     * @param features Özellik vektörü
     * @param label    Sınıf etiketi (+1 veya -1)
     * @throws IllegalArgumentException geçersiz label veya boş features
     */
    public DataPoint(double[] features, int label) {
        if (features == null || features.length == 0) {
            throw new IllegalArgumentException("Features boş olamaz.");
        }
        if (label != 1 && label != -1) {
            throw new IllegalArgumentException("Label yalnızca +1 veya -1 olabilir. Verilen: " + label);
        }

        this.features = Arrays.copyOf(features, features.length);
        this.label = label;
    }

    /**
     * @return Özellik vektörünün kopyası
     */
    public double[] getFeatures() {
        return Arrays.copyOf(features, features.length);
    }

    /**
     * @return Boyut sayısı
     */
    public int getDimension() {
        return features.length;
    }

    /**
     * Belirli bir indeksteki özelliği döndürür
     */
    public double getFeature(int index) {
        if (index < 0 || index >= features.length) {
            throw new IndexOutOfBoundsException("Geçersiz indeks: " + index);
        }
        return features[index];
    }

    public int getLabel() {
        return label;
    }

    /**
     * İki veri noktası arasındaki dot product
     */
    public double dotProduct(DataPoint other) {
        if (this.features.length != other.features.length) {
            throw new IllegalArgumentException("Boyut uyumsuzluğu.");
        }
        double sum = 0.0;
        for (int i = 0; i < features.length; i++) {
            sum += this.features[i] * other.features[i];
        }
        return sum;
    }

    /**
     * Bu nokta ile verilen ağırlık vektörü arasındaki dot product
     */
    public double dotProduct(double[] weights) {
        if (weights.length != features.length) {
            throw new IllegalArgumentException("Boyut uyumsuzluğu.");
        }
        double sum = 0.0;
        for (int i = 0; i < features.length; i++) {
            sum += features[i] * weights[i];
        }
        return sum;
    }

    @Override
    public String toString() {
        return String.format("DataPoint{features=%s, label=%+d}",
                Arrays.toString(features), label);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof DataPoint)) return false;
        DataPoint that = (DataPoint) o;
        return label == that.label && Arrays.equals(features, that.features);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(label);
        result = 31 * result + Arrays.hashCode(features);
        return result;
    }
}