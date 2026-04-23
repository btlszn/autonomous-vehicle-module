package data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Veri setini yöneten sınıf.
 * DataPoint listesini güvenli şekilde saklar ve sorgular.
 */
public final class Dataset {

    private final List<DataPoint> points;

    public Dataset() {
        this.points = new ArrayList<>();
    }

    /**
     * Veri noktası ekler
     */
    public void add(DataPoint point) {
        if (point == null) {
            throw new IllegalArgumentException("DataPoint null olamaz.");
        }
        points.add(point);
    }

    /**
     * @return Değiştirilemez liste
     */
    public List<DataPoint> getPoints() {
        return Collections.unmodifiableList(points);
    }

    /**
     * Belirli sınıfa ait noktaları döndürür
     */
    public List<DataPoint> getPointsByLabel(int label) {
        return points.stream()
                .filter(p -> p.getLabel() == label)
                .collect(Collectors.toList());
    }

    public int size() {
        return points.size();
    }

    public boolean isEmpty() {
        return points.isEmpty();
    }

    /**
     * Veri setinin boyutunu döndürür (feature sayısı)
     */
    public int getDimension() {
        if (points.isEmpty()) return 0;
        return points.get(0).getDimension();
    }

    /**
     * İki farklı sınıf var mı kontrol eder
     */
    public boolean hasBothClasses() {
        boolean hasPos = points.stream().anyMatch(p -> p.getLabel() == 1);
        boolean hasNeg = points.stream().anyMatch(p -> p.getLabel() == -1);
        return hasPos && hasNeg;
    }

    public void printSummary() {
        long posCount = points.stream().filter(p -> p.getLabel() == 1).count();
        long negCount = points.stream().filter(p -> p.getLabel() == -1).count();
        System.out.printf("[INFO] Toplam: %d nokta | Sınıf +1 (Tehlikeli): %d | Sınıf -1 (Güvenli): %d%n%n",
                points.size(), posCount, negCount);
    }
}
