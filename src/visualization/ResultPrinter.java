package visualization;

import data.DataPoint;
import model.SVMModel;

import java.util.Arrays;
import java.util.List;

/**
 * SVM sonuçlarını konsola yazdıran görselleştirici.
 *
 * Sorumluluk: Yalnızca görselleştirme / çıktı.
 * Hesaplama yapmaz - Single Responsibility Principle.
 */
public class ResultPrinter {

    private static final String SEPARATOR = "─".repeat(56);
    private static final String DOUBLE_SEP = "═".repeat(56);

    /**
     * Eğitilmiş modelin özetini yazdırır
     */
    public void printModel(SVMModel model) {
        System.out.println("\n" + DOUBLE_SEP);
        System.out.println("  EĞITIM SONUÇLARI");
        System.out.println(DOUBLE_SEP);

        // Ağırlık vektörü
        double[] w = model.getWeights();
        System.out.printf("  Ağırlık vektörü (w): %s%n", Arrays.toString(formatArray(w)));

        // Bias
        System.out.printf("  Bias (b):             %.6f%n", model.getBias());

        // Norm ve marjin
        System.out.printf("  ||w||:                %.6f%n", model.getWeightNorm());
        System.out.printf("  Marjin (2/||w||):     %.6f  ← MAKSIMUM%n", model.getMargin());

        // Destek vektörleri
        List<DataPoint> svs = model.getSupportVectors();
        System.out.printf("%n  Destek Vektörleri (%d adet):%n", svs.size());
        for (DataPoint sv : svs) {
            System.out.printf("    %s%n", sv);
        }

        System.out.println(DOUBLE_SEP);
    }

    /**
     * Karar sınırı denklemini yazdırır
     */
    public void printDecisionBoundary(SVMModel model) {
        double[] w = model.getWeights();
        double b = model.getBias();

        System.out.println("\n" + SEPARATOR);
        System.out.println("  KARAR SINIRI DENKLEMİ");
        System.out.println(SEPARATOR);

        if (w.length == 2) {
            // 2D: w1*x1 + w2*x2 + b = 0  =>  x2 = (-w1*x1 - b) / w2
            System.out.printf("  Hiperplane: %.4f·x₁ + %.4f·x₂ + (%.4f) = 0%n",
                    w[0], w[1], b);

            if (Math.abs(w[1]) > 1e-10) {
                System.out.printf("  Doğrusal form: x₂ = %.4f·x₁ + %.4f%n",
                        -w[0] / w[1], -b / w[1]);
            }

            System.out.println();
            System.out.printf("  Pozitif marjin: %.4f·x₁ + %.4f·x₂ + (%.4f) = +1%n",
                    w[0], w[1], b);
            System.out.printf("  Negatif marjin: %.4f·x₁ + %.4f·x₂ + (%.4f) = -1%n",
                    w[0], w[1], b);

            // ASCII görselleştirme
            printAsciiGrid(model);
        } else {
            // Genel boyut
            StringBuilder sb = new StringBuilder("  Hiperplane: ");
            for (int i = 0; i < w.length; i++) {
                sb.append(String.format("%.4f·x%d ", w[i], i + 1));
                if (i < w.length - 1) sb.append("+ ");
            }
            sb.append(String.format("+ (%.4f) = 0", b));
            System.out.println(sb);
        }

        System.out.println(SEPARATOR);
    }

    /**
     * Zaman karmaşıklığı analizini yazdırır
     */
    public void printComplexityAnalysis() {
        System.out.println("\n" + SEPARATOR);
        System.out.println("  ZAMAN KARMAŞIKLIĞI ANALİZİ");
        System.out.println(SEPARATOR);
        System.out.println("  Algoritma: SMO (Sequential Minimal Optimization)");
        System.out.println();
        System.out.println("  Adım                        Karmaşıklık");
        System.out.println("  " + "─".repeat(52));
        System.out.println("  Kernel matrisi (Gram)        O(n² · d)");
        System.out.println("  SMO iterasyonu (beklenen)    O(n² ~ n³)");
        System.out.println("  Ağırlık hesabı               O(n · d)");
        System.out.println("  Tahmin (predict)             O(d)");
        System.out.println("  " + "─".repeat(52));
        System.out.println("  n = örnek sayısı, d = boyut sayısı");
        System.out.println();
        System.out.println("  NEDEN MAXIMUM MARGIN OPTİMAL?");
        System.out.println("  Structural Risk Minimization (Vapnik, 1995):");
        System.out.println("  Daha geniş marjin => daha düşük VC boyutu");
        System.out.println("     => daha iyi genelleme garantisi");
        System.out.println(SEPARATOR);
    }

    /**
     * 2D veri için ASCII ızgara görselleştirme
     */
    private void printAsciiGrid(SVMModel model) {
        System.out.println("\n  ASCII Görselleştirme (yaklaşık):");
        System.out.println("  (+ = Tehlikeli Engel, - = Güvenli Bölge, | = Karar Sınırı)\n");

        int rows = 8, cols = 20;
        double xMin = 0, xMax = 10, yMin = 0, yMax = 8;
        char[][] grid = new char[rows][cols];
        for (char[] row : grid) Arrays.fill(row, ' ');

        double[] w = model.getWeights();
        double b = model.getBias();

        for (int col = 0; col < cols; col++) {
            double x1 = xMin + (xMax - xMin) * col / cols;
            for (int row = 0; row < rows; row++) {
                double x2 = yMax - (yMax - yMin) * row / rows;
                double score = w[0] * x1 + w[1] * x2 + b;
                if (Math.abs(score) < 0.8) {
                    grid[row][col] = '|';
                } else if (score > 0) {
                    grid[row][col] = '+';
                } else {
                    grid[row][col] = '-';
                }
            }
        }

        for (char[] row : grid) {
            System.out.print("  ");
            System.out.println(new String(row));
        }
        System.out.println();
    }

    private double[] formatArray(double[] arr) {
        double[] formatted = new double[arr.length];
        for (int i = 0; i < arr.length; i++) {
            formatted[i] = Math.round(arr[i] * 10000.0) / 10000.0;
        }
        return formatted;
    }
}