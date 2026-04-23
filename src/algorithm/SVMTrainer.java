package algorithm;

import data.DataPoint;
import data.Dataset;
import model.SVMModel;

import java.util.ArrayList;
import java.util.List;

/**
 * SVM Eğitim Orkestratörü
 */
public class SVMTrainer {

    private static final double ALPHA_THRESHOLD = 1e-5;

    /**
     * Veri setini eğitir ve model döndürür.
     *
     * @param dataset Eğitim veri seti
     * @return Eğitilmiş SVMModel
     * @throws IllegalStateException Veri yetersiz veya ayrılamıyorsa
     */
    public SVMModel train(Dataset dataset) {
        System.out.println(" SVM başlıyor...");

        // 1. Doğrulama
        validateDataset(dataset);

        List<DataPoint> points = dataset.getPoints();

        // 2. Dual problem çözümü (SMO)
        System.out.println(" SMO algoritması çalışıyor...");
        QuadraticProgrammingSolver solver = new QuadraticProgrammingSolver(points);
        solver.solve();

        double[] alphas = solver.getAlphas();
        double bias = solver.getBias();

        // 3. Destek vektörlerini topla
        List<DataPoint> supportVectors = new ArrayList<>();
        for (int i = 0; i < points.size(); i++) {
            if (alphas[i] > ALPHA_THRESHOLD) {
                supportVectors.add(points.get(i));
            }
        }

        if (supportVectors.isEmpty()) {
            throw new IllegalStateException("Destek vektörü bulunamadı - veri ayrılamıyor olabilir.");
        }

        System.out.printf("[TRAIN] %d destek vektörü bulundu.%n", supportVectors.size());

        // 4. Primal ağırlık vektörü: w = Σ α_i y_i x_i
        int dim = dataset.getDimension();
        double[] weights = computeWeights(alphas, points, dim);

        // 5. Marjin: 2 / ||w||
        double weightNorm = MathUtils.norm(weights);
        double margin = 2.0 / weightNorm;

        System.out.printf("[TRAIN] ||w|| = %.6f, Marjin = %.6f%n", weightNorm, margin);

        return new SVMModel(weights, bias, margin, supportVectors);
    }

    /**
     * Primal ağırlık vektörünü hesaplar.
     * Zaman Karmaşıklığı: O(n * d)
     */
    private double[] computeWeights(double[] alphas, List<DataPoint> points, int dim) {
        double[] weights = new double[dim];
        for (int i = 0; i < points.size(); i++) {
            if (alphas[i] > ALPHA_THRESHOLD) {
                double coeff = alphas[i] * points.get(i).getLabel();
                double[] xi = points.get(i).getFeatures();
                for (int d = 0; d < dim; d++) {
                    weights[d] += coeff * xi[d];
                }
            }
        }
        return weights;
    }

    /**
     * Veri setinin eğitime hazır olduğunu doğrular.
     */
    private void validateDataset(Dataset dataset) {
        if (dataset == null || dataset.isEmpty()) {
            throw new IllegalArgumentException("Veri seti boş olamaz.");
        }
        if (!dataset.hasBothClasses()) {
            throw new IllegalArgumentException("Veri seti hem +1 hem -1 sınıfını içermelidir.");
        }
        if (dataset.size() < 2) {
            throw new IllegalArgumentException("En az 2 veri noktası gereklidir.");
        }
    }
}
