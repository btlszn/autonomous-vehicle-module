package algorithm;

import java.util.Arrays;

/**
 * SVM hesaplamalarında kullanılan matematiksel yardımcı sınıf.
 * Tüm metodlar statik - utility class pattern.
 *
 * Kapsanan işlemler:
 * - Vektör işlemleri (dot product, norm, skalar çarpım)
 * - Matris işlemleri (2x2 sistem çözümü)
 * - Geometrik hesaplamalar (projeksiyon, mesafe)
 */
public final class MathUtils {

    // Utility class - instantiate edilemez
    private MathUtils() {
        throw new UnsupportedOperationException("Utility class");
    }

    // ========================
    // Vektör İşlemleri
    // ========================

    /**
     * İki vektör arasındaki dot product: a · b
     * Zaman Karmaşıklığı: O(n)
     */
    public static double dotProduct(double[] a, double[] b) {
        validateSameLength(a, b);
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    /**
     * Vektör normu (L2): ||v|| = sqrt(Σv_i²)
     * Zaman Karmaşıklığı: O(n)
     */
    public static double norm(double[] v) {
        double sumSq = 0.0;
        for (double val : v) {
            sumSq += val * val;
        }
        return Math.sqrt(sumSq);
    }

    /**
     * Vektör normunun karesi: ||v||²
     * Zaman Karmaşıklığı: O(n)
     */
    public static double normSquared(double[] v) {
        double sumSq = 0.0;
        for (double val : v) {
            sumSq += val * val;
        }
        return sumSq;
    }

    /**
     * Vektörü skalar ile çarpar: c * v
     * Zaman Karmaşıklığı: O(n)
     */
    public static double[] scalarMultiply(double scalar, double[] v) {
        double[] result = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            result[i] = scalar * v[i];
        }
        return result;
    }

    /**
     * İki vektörü toplar: a + b
     * Zaman Karmaşıklığı: O(n)
     */
    public static double[] add(double[] a, double[] b) {
        validateSameLength(a, b);
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] + b[i];
        }
        return result;
    }

    /**
     * İki vektörü çıkarır: a - b
     * Zaman Karmaşıklığı: O(n)
     */
    public static double[] subtract(double[] a, double[] b) {
        validateSameLength(a, b);
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] - b[i];
        }
        return result;
    }

    /**
     * Vektörü normalize eder: v / ||v||
     * Zaman Karmaşıklığı: O(n)
     */
    public static double[] normalize(double[] v) {
        double n = norm(v);
        if (n < 1e-12) {
            throw new ArithmeticException("Sıfır vektör normalize edilemez.");
        }
        return scalarMultiply(1.0 / n, v);
    }

    // ========================
    // Matris İşlemleri
    // ========================

    /**
     * 2x2 lineer sistemi çözer: Ax = b
     * Cramer's Rule kullanılır.
     * | a11 a12 | |x1|   |b1|
     * | a21 a22 | |x2| = |b2|
     *
     * Zaman Karmaşıklığı: O(1)
     */
    public static double[] solve2x2(double a11, double a12,
                                    double a21, double a22,
                                    double b1, double b2) {
        double det = a11 * a22 - a12 * a21;
        if (Math.abs(det) < 1e-12) {
            throw new ArithmeticException("Singular matris - sistem çözülemiyor.");
        }
        double x1 = (b1 * a22 - b2 * a12) / det;
        double x2 = (a11 * b2 - a21 * b1) / det;
        return new double[]{x1, x2};
    }

    /**
     * Gauss eliminasyonu ile nxn lineer sistem çözer: Ax = b
     * Pivoting ile sayısal kararlılık sağlanır.
     *
     * Zaman Karmaşıklığı: O(n³)
     */
    public static double[] gaussianElimination(double[][] A, double[] b) {
        int n = b.length;
        // Augmented matrix oluştur
        double[][] aug = new double[n][n + 1];
        for (int i = 0; i < n; i++) {
            System.arraycopy(A[i], 0, aug[i], 0, n);
            aug[i][n] = b[i];
        }

        // Forward elimination (partial pivoting ile)
        for (int col = 0; col < n; col++) {
            // Pivot satırı bul
            int pivotRow = col;
            for (int row = col + 1; row < n; row++) {
                if (Math.abs(aug[row][col]) > Math.abs(aug[pivotRow][col])) {
                    pivotRow = row;
                }
            }
            // Satırları değiştir
            double[] temp = aug[col];
            aug[col] = aug[pivotRow];
            aug[pivotRow] = temp;

            if (Math.abs(aug[col][col]) < 1e-12) {
                throw new ArithmeticException("Singular matris - eliminasyon başarısız.");
            }

            // Eliminasyon
            for (int row = col + 1; row < n; row++) {
                double factor = aug[row][col] / aug[col][col];
                for (int j = col; j <= n; j++) {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }

        // Back substitution
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            x[i] = aug[i][n];
            for (int j = i + 1; j < n; j++) {
                x[i] -= aug[i][j] * x[j];
            }
            x[i] /= aug[i][i];
        }
        return x;
    }

    // ========================
    // Geometrik Hesaplamalar
    // ========================

    /**
     * Bir noktanın hiperplane'e olan işaretli mesafesi
     * Hiperplane: w·x + b = 0
     *
     * Zaman Karmaşıklığı: O(n)
     */
    public static double signedDistance(double[] point, double[] w, double b) {
        return (dotProduct(w, point) + b) / norm(w);
    }

    /**
     * İki nokta arasındaki Öklid mesafesi
     * Zaman Karmaşıklığı: O(n)
     */
    public static double euclideanDistance(double[] a, double[] b) {
        validateSameLength(a, b);
        double sumSq = 0.0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sumSq += diff * diff;
        }
        return Math.sqrt(sumSq);
    }

    // ========================
    // Validasyon Yardımcıları
    // ========================

    private static void validateSameLength(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException(
                    String.format("Vektör boyutları uyuşmuyor: %d != %d", a.length, b.length));
        }
    }
}
