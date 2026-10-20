/* One-loop t tbar H test functions f2 (q qbar) and f4 (g g) of arXiv:2412.09534 from a
 * GoSam OLP library (the same phase-space parametrisation as the authors' tth-eval.cpp).
 *
 * stdin:  lines "beta2 frac_stt theta_h theta_t phi_t"
 * stdout: "weight tree finite pole1 pole2 cc_ttbar beta_ttbar accuracy" per line, where
 *   tree      = |M0|^2 (couplings stripped off: g_s = e = 1),
 *   finite    = 2 Re <M0|M1> finite part, DRED, mu_R = sqrt(s)/2, no alpha_s/(2 pi) factor,
 *   pole1/2   = coefficients of 1/eps and 1/eps^2 in the same units,
 *   cc_ttbar  = <M0| T_t . T_tbar |M0> (colour-correlated Born),
 *   beta_ttbar= velocity of the tops in the t tbar rest frame,
 *   accuracy  = GoSam's estimate of the relative precision of the loop amplitude.
 * The paper's a2/a4 is then finite + pi^2 / beta_ttbar * cc_ttbar (eq. 14/19).
 * Build with -DPREFIX=tthqq_ or -DPREFIX=tthgg_ against the matching OLP library.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#define CAT2(a, b) a##b
#define CAT(a, b) CAT2(a, b)
#define OLP(name) CAT(PREFIX, name)

extern "C" {
    void OLP(OLP_Start)(const char* contract_file_name, int* ierr);
    void OLP(OLP_Option)(const char* line, int* stat);
    void OLP(OLP_EvalSubProcess2)(int* label, double* momenta, double* mu, double* res, double* acc);
    void OLP(OLP_Finalize)();
}

static double sqr(double x) { return x * x; }
static double range(double a, double b, double x) { return a + x * (b - a); }
static double kallen(double a, double sqrt_b, double sqrt_c) {
    return (a - sqr(sqrt_b + sqrt_c)) * (a - sqr(sqrt_b - sqrt_c));
}
static void transform(double x[4], const double m[4][4]) {
    double y[4];
    for (int i = 0; i < 4; i++) y[i] = m[i][0] * x[0] + m[i][1] * x[1] + m[i][2] * x[2] + m[i][3] * x[3];
    memcpy(x, y, sizeof(y));
}
static void option(const char* line) {
    int stat = 0;
    OLP(OLP_Option)(line, &stat);
    if (stat != 1) { fprintf(stderr, "OLP_Option(%s) failed: %d\n", line, stat); }
}

int main(int argc, char* argv[]) {
    if (argc != 2) { fprintf(stderr, "usage: %s <contract file>\n", argv[0]); return 1; }
    option("nlo_prefactors=0");
    option("mT=1");
    option("mH=0.7223151185146152");
    // keep GoSam's own messages (Fortran unit 6) out of the result stream
    FILE* fout = fdopen(dup(1), "w");
    fclose(stdout);
    int ierr = 0;
    OLP(OLP_Start)(argv[1], &ierr);
    if (ierr != 1) fprintf(stderr, "OLP_Start returned %d\n", ierr);
    const double mt = 1.0, mh = 0.7223151185146152;      // sqrt(12/23) m_t
    double beta2, frac_stt, theta_h, theta_t, phi_t;
    while (scanf(" %lf %lf %lf %lf %lf", &beta2, &frac_stt, &theta_h, &theta_t, &phi_t) == 5) {
        double s = sqr(2 * mt + mh) / (1 - beta2), sqrt_s = sqrt(s);
        double stt = range(4 * mt * mt, sqr(sqrt_s - mh), frac_stt), sqrt_stt = sqrt(stt);
        double pt = sqrt(stt - 4 * mt * mt) / 2;
        double qt1[4] = {sqrt_stt / 2, pt * sin(theta_t) * cos(phi_t), pt * sin(theta_t) * sin(phi_t), pt * cos(theta_t)};
        double qt2[4] = {qt1[0], -qt1[1], -qt1[2], -qt1[3]};
        double kallen_stt_s_mh2 = kallen(stt, sqrt_s, mh);
        double qh2 = kallen_stt_s_mh2 / 4 / s, qh = sqrt(qh2), Eh = sqrt(qh2 + mh * mh);
        double invrot[4][4] = {{1, 0, 0, 0}, {0, -cos(theta_h), 0, -sin(theta_h)}, {0, 0, 1, 0}, {0, sin(theta_h), 0, -cos(theta_h)}};
        double g = sqrt(1 + qh2 / stt), gb = qh / sqrt_stt;
        double invboost[4][4] = {{g, 0, 0, gb}, {0, 1, 0, 0}, {0, 0, 1, 0}, {gb, 0, 0, g}};
        double q1[4] = {sqrt_s / 2, 0, 0, sqrt_s / 2}, q2[4] = {sqrt_s / 2, 0, 0, -sqrt_s / 2};
        double qH[4] = {Eh, qh * sin(theta_h), 0, qh * cos(theta_h)};
        transform(qt1, invboost); transform(qt2, invboost);
        transform(qt1, invrot); transform(qt2, invrot);
        double beta_tt = 2 * pt / sqrt_stt;
        double weight = (sqr(sqrt_s - mh) - 4 * mt * mt) * sqrt(stt * (stt - 4 * mt * mt) * kallen_stt_s_mh2)
                        / (1024 * pow(M_PI, 4) * s * stt) * sin(theta_h) * sin(theta_t);
        // legs: parton, parton, H, t, tbar; five numbers per leg (E, px, py, pz, m)
        double momenta[50] = {0};
        const double* legs[5] = {q1, q2, qH, qt1, qt2};
        const double masses[5] = {0, 0, mh, mt, mt};
        for (int i = 0; i < 5; i++) { memcpy(&momenta[5 * i], legs[i], 4 * sizeof(double)); momenta[5 * i + 4] = masses[i]; }
        double mu = sqrt_s / 2, res[60] = {0}, cc[60] = {0}, acc = NAN, acc_cc = NAN;
        int label = 0;
        OLP(OLP_EvalSubProcess2)(&label, momenta, &mu, res, &acc);
        label = 1;
        OLP(OLP_EvalSubProcess2)(&label, momenta, &mu, cc, &acc_cc);
        // ccTree entries are ordered by pairs (i<j) with index (i-1) + (j-1)(j-2)/2; the tops are legs 4 and 5
        fprintf(fout, "%.16e %.16e %.16e %.16e %.16e %.16e %.16e %.3e\n", weight, res[3], res[2], res[1], res[0], cc[9], beta_tt, acc);
        fflush(fout);
    }
    OLP(OLP_Finalize)();
    return 0;
}
