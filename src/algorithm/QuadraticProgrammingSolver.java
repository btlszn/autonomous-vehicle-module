package algorithm;

import data.DataPoint;
import java.util.List;

/**
 * SVM Dual Problem için Quadratic Programming çözücüsü.
 * Algoritma: Sequential Minimal Optimization (SMO) - John Platt (1998)
 * Zaman Karmaşıklığı: O(n²) ~ O(n³) iterasyon başına
 */
public class QuadraticProgrammingSolver {

    private static final double TOLERANCE = 1e-5;  // KKT toleransı
    private static final int MAX_ITERATIONS = 10000;
    private static final double EPS = 1e-8;         // Sayısal sıfır

    private final List<DataPoint> dataPoints;
    private final int n;
    private final double[] alpha;   // Lagrange çarpanları
    private double bias;            // Bias terimi

    // Kernel matrisi (gram matrix) - önceden hesaplanır: O(n²) bellek
    private final double[][] kernelMatrix;

    public QuadraticProgrammingSolver(List<DataPoint> dataPoints) {
        this.dataPoints = dataPoints;
        this.n = dataPoints.size();
        this.alpha = new double[n];     // Sıfır başlatma
        this.bias = 0.0;
        this.kernelMatrix = precomputeKernelMatrix();
    }

    /**
     * SMO algoritmasını çalıştırır ve alpha değerlerini bulur.
     *
     * SMO - Sequential Minimal Optimization:
     * Her adımda 2 Lagrange çarpanı seçilir ve analitik olarak optimize edilir.
     * Bu sayede büyük QP problemi çözülmeden iteratif yakınsama sağlanır.
     *
     * Zaman Karmaşıklığı:
     *   - Kernel matrisi önceden hesaplama: O(n²)
     *   - Her iterasyon:                    O(n)
     *   - Toplam beklenen:                  O(n² ~ n³)
     */
    public void solve() {
        int iterations = 0;
        boolean changed = true;

        while (changed && iterations < MAX_ITERATIONS) {
            changed = false;
            iterations++;

            // Her alpha için KKT ihlali kontrol et
            for (int i = 0; i < n; i++) {
                double Ei = computeError(i);
                double yi = dataPoints.get(i).getLabel();

                // KKT koşulu ihlali var mı?
                if (kktViolation(yi, Ei)) {
                    // İkinci alpha'yı seç (max |Ei - Ej| heuristic)
                    int j = selectSecondAlpha(i, Ei);
                    if (j < 0) continue;

                    if (optimizePair(i, j)) {
                        changed = true;
                    }
                }
            }
        }

        System.out.printf("[SMO] %d iterasyonda yakınsandı.%n", iterations);
    }

    /**
     * İki alpha'yı analitik olarak optimize eder (SMO'nun çekirdeği)
     *
     * Analitik çözüm:
     *   η = K(x_i, x_i) + K(x_j, x_j) - 2K(x_i, x_j)
     *   α_j^new = α_j + y_j(E_i - E_j) / η
     *   α_j^new = clip(α_j^new, L, H)
     *   α_i^new = α_i + y_i y_j (α_j - α_j^new)
     */
    private boolean optimizePair(int i, int j) {
        if (i == j) return false;

        DataPoint pi = dataPoints.get(i);
        DataPoint pj = dataPoints.get(j);
        double yi = pi.getLabel();
        double yj = pj.getLabel();

        double Ei = computeError(i);
        double Ej = computeError(j);

        double alphaIOld = alpha[i];
        double alphaJOld = alpha[j];

        // Box constraints (L, H): hard-margin SVM için basit clip
        double L, H;
        if (yi != yj) {
            L = Math.max(0, alpha[j] - alpha[i]);
            H = Double.MAX_VALUE; // Hard-margin: sınırsız üst
        } else {
            L = Math.max(0, alpha[i] + alpha[j]);
            H = Double.MAX_VALUE;
        }

        // Kernel değerleri kernel matrisinden alınır: O(1)
        double Kii = kernelMatrix[i][i];
        double Kjj = kernelMatrix[j][j];
        double Kij = kernelMatrix[i][j];

        // η: ikinci dereceden terim katsayısı
        double eta = Kii + Kjj - 2.0 * Kij;
        if (eta <= EPS) return false;

        // α_j güncelleme
        alpha[j] += yj * (Ei - Ej) / eta;
        alpha[j] = Math.max(L, alpha[j]);

        if (Math.abs(alpha[j] - alphaJOld) < EPS) return false;

        // α_i güncelleme (denklem kısıtından türetilir)
        alpha[i] += yi * yj * (alphaJOld - alpha[j]);
        alpha[i] = Math.max(0, alpha[i]);

        // Bias güncelleme
        updateBias(i, j, Ei, Ej, alphaIOld, alphaJOld);

        return true;
    }

    /**
     * KKT koşulunun ihlal edilip edilmediğini kontrol eder
     * KKT: y_i * f(x_i) >= 1 için α_i = 0 (marjin dışı)
     *      y_i * f(x_i) = 1 için 0 < α_i (destek vektörü)
     */
    private boolean kktViolation(double yi, double Ei) {
        return (yi * Ei < -TOLERANCE) || (yi * Ei > TOLERANCE);
    }

    /**
     * İkinci alpha'yı seçer: maksimum |Ei - Ej| heuristic
     * Zaman Karmaşıklığı: O(n)
     */
    private int selectSecondAlpha(int i, double Ei) {
        double maxDiff = 0.0;
        int bestJ = -1;
        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            double Ej = computeError(j);
            double diff = Math.abs(Ei - Ej);
            if (diff > maxDiff) {
                maxDiff = diff;
                bestJ = j;
            }
        }
        return bestJ;
    }

    /**
     * f(x_i) - y_i hatası
     * Zaman Karmaşıklığı: O(n) (kernel matrisinden O(1) lookup)
     */
    private double computeError(int i) {
        return decisionFunction(i) - dataPoints.get(i).getLabel();
    }

    /**
     * Karar fonksiyonu: f(x_i) = Σ_j α_j y_j K(x_j, x_i) + b
     * Zaman Karmaşıklığı: O(n)
     */
    private double decisionFunction(int i) {
        double sum = bias;
        for (int j = 0; j < n; j++) {
            if (alpha[j] > EPS) {
                sum += alpha[j] * dataPoints.get(j).getLabel() * kernelMatrix[j][i];
            }
        }
        return sum;
    }

    /**
     * Bias güncelleme (KKT'den türetilir)
     */
    private void updateBias(int i, int j, double Ei, double Ej,
                            double alphaIOld, double alphaJOld) {
        double yi = dataPoints.get(i).getLabel();
        double yj = dataPoints.get(j).getLabel();

        double b1 = bias - Ei
                - yi * (alpha[i] - alphaIOld) * kernelMatrix[i][i]
                - yj * (alpha[j] - alphaJOld) * kernelMatrix[i][j];

        double b2 = bias - Ej
                - yi * (alpha[i] - alphaIOld) * kernelMatrix[i][j]
                - yj * (alpha[j] - alphaJOld) * kernelMatrix[j][j];

        bias = (b1 + b2) / 2.0;
    }

    /**
     * Gram matrisini (kernel matrisi) önceden hesaplar.
     * K[i][j] = x_i · x_j (lineer kernel)
     *
     * Zaman Karmaşıklığı: O(n² * d)
     * Bellek: O(n²)
     */
    private double[][] precomputeKernelMatrix() {
        double[][] K = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                double val = dataPoints.get(i).dotProduct(dataPoints.get(j));
                K[i][j] = val;
                K[j][i] = val; // Simetrik
            }
        }
        return K;
    }

    // --- Erişimciler ---

    public double[] getAlphas() {
        return alpha.clone();
    }

    public double getBias() {
        return bias;
    }

    /**
     * Destek vektörü olup olmadığını kontrol eder
     * α_i > EPS ise destek vektörü
     */
    public boolean isSupportVector(int index) {
        return alpha[index] > EPS;
    }
}