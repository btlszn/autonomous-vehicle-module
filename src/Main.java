package main;

import algorithm.SVMTrainer;
import data.DataPoint;
import data.Dataset;
import model.SVMModel;
import visualization.ResultPrinter;

import java.util.List;

/**
 * Otonom Araç Navigasyon Sistemi - Güvenlik Modülü
 * SVM (Support Vector Machine) ile Engel Sınıflandırma
 *
 * Bu program, iki boyutlu uzayda iki farklı engel sınıfını
 * maksimum marjin ile ayıran optimal karar sınırını bulur.
 */
public class Main {

    public static void main(String[] args) {
        System.out.println("=================================================");
        System.out.println("  Otonom Araç Navigasyon - Güvenlik Modülü");
        System.out.println("  Hard-Margin SVM Implementasyonu");
        System.out.println("=================================================\n");

        // Veri setini oluştur
        Dataset dataset = buildSampleDataset();

        System.out.println(" Veri seti yüklendi: " + dataset.size() + " nokta");
        dataset.printSummary();

        // SVM eğitimi
        SVMTrainer trainer = new SVMTrainer();
        SVMModel model = trainer.train(dataset);

        // Sonuçları yazdır
        ResultPrinter printer = new ResultPrinter();
        printer.printModel(model);
        printer.printDecisionBoundary(model);
        printer.printComplexityAnalysis();

        // Örnek tahminler
        runPredictions(model);

        System.out.println("\n Program başarıyla tamamlandı.");
    }

    /**
     * Örnek veri seti - iki engel sınıfı
     * Sınıf +1: Tehlikeli Engeller
     * Sınıf -1: Güvenli Bölgeler
     */
    private static Dataset buildSampleDataset() {
        Dataset dataset = new Dataset();

        // Sınıf +1 (Tehlikeli Engeller)
        dataset.add(new DataPoint(new double[]{1.0, 2.0}, 1));
        dataset.add(new DataPoint(new double[]{2.0, 3.0}, 1));
        dataset.add(new DataPoint(new double[]{3.0, 3.0}, 1));
        dataset.add(new DataPoint(new double[]{2.0, 4.0}, 1));
        dataset.add(new DataPoint(new double[]{4.0, 2.0}, 1));

        // Sınıf -1 (Güvenli Bölgeler)
        dataset.add(new DataPoint(new double[]{6.0, 5.0}, -1));
        dataset.add(new DataPoint(new double[]{7.0, 6.0}, -1));
        dataset.add(new DataPoint(new double[]{8.0, 5.0}, -1));
        dataset.add(new DataPoint(new double[]{7.0, 4.0}, -1));
        dataset.add(new DataPoint(new double[]{9.0, 6.0}, -1));

        return dataset;
    }

    /**
     * Eğitilmiş model ile yeni noktalarda tahmin yap
     */
    private static void runPredictions(SVMModel model) {
        System.out.println("\n--- Tahmin Testleri ---");

        double[][] testPoints = {
                {3.0, 5.0},
                {6.0, 3.0},
                {4.5, 4.5},
                {1.0, 1.0},
                {9.0, 9.0}
        };

        for (double[] point : testPoints) {
            int prediction = model.predict(point);
            double score = model.decisionScore(point);
            String label = prediction == 1 ? "TEHLİKELİ ENGEL" : "GÜVENLİ BÖLGE";
            System.out.printf("  Nokta (%.1f, %.1f) -> %s (skor: %.4f)%n",
                    point[0], point[1], label, score);
        }
    }
}