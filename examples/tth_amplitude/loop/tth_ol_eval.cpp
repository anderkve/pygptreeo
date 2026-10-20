/* One-loop t tbar H amplitudes from OpenLoops, in the layout of tth_loop_eval.cpp.
 *
 * usage: tth_ol_eval <qq|gg> [--cc]   reads "beta2 frac_stt theta_h theta_t phi_t" lines,
 * writes "weight tree finite pole1 pole2 cc_ttbar beta_ttbar accuracy" per line.
 * tree, finite and the poles are in OpenLoops' normalisation divided by alpha_s/(2 pi)
 * for the loop part (so pole2/tree = -2 C_F for q qbar, -2 C_A for g g), 't Hooft-Veltman
 * scheme, on-shell top mass, 5 massless flavours, mu_R = sqrt(s)/2, m_t = 173 GeV.
 * With --cc the colour-correlated Born of every pair is printed instead (one line).
 */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "openloops.h"

static double sqr(double x) { return x * x; }
static double range(double a, double b, double x) { return a + x * (b - a); }
static double kallen(double a, double b, double c) { return (a - sqr(b + c)) * (a - sqr(b - c)); }
static void transform(double x[4], const double m[4][4]) { double y[4]; for (int i = 0; i < 4; i++) y[i] = m[i][0]*x[0]+m[i][1]*x[1]+m[i][2]*x[2]+m[i][3]*x[3]; memcpy(x, y, sizeof y); }

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <qq|gg> [--cc]\n", argv[0]); return 1; }
    int gg = strcmp(argv[1], "gg") == 0, want_cc = argc > 2 && strcmp(argv[2], "--cc") == 0;
    const double MT = 173.0, mt = 1.0, mh = 0.7223151185146152;
    ol_setparameter_int("verbose", 0);
    ol_setparameter_int("order_ew", 1);
    ol_setparameter_double("mass(6)", MT);
    ol_setparameter_double("width(6)", 0.0);
    ol_setparameter_double("mass(25)", MT * mh);
    ol_setparameter_double("width(25)", 0.0);
    ol_setparameter_double("mass(5)", 0.0);
    const double alphas = 0.118;
    ol_setparameter_double("alphas", alphas);
    int id = ol_register_process(gg ? "21 21 -> 6 -6 25" : "2 -2 -> 6 -6 25", 11);
    if (id <= 0) { fprintf(stderr, "process registration failed: %d\n", id); return 1; }
    ol_start();
    FILE* fout = stdout;
    double beta2, frac, th, tt, ph;
    while (scanf(" %lf %lf %lf %lf %lf", &beta2, &frac, &th, &tt, &ph) == 5) {
        double s = sqr(2 * mt + mh) / (1 - beta2), rs = sqrt(s);
        double stt = range(4 * mt * mt, sqr(rs - mh), frac), rstt = sqrt(stt), pt = sqrt(stt - 4 * mt * mt) / 2;
        double qt1[4] = {rstt / 2, pt * sin(tt) * cos(ph), pt * sin(tt) * sin(ph), pt * cos(tt)}, qt2[4] = {qt1[0], -qt1[1], -qt1[2], -qt1[3]};
        double kl = kallen(stt, rs, mh), qh2 = kl / 4 / s, qh = sqrt(qh2), Eh = sqrt(qh2 + mh * mh);
        double invrot[4][4] = {{1,0,0,0},{0,-cos(th),0,-sin(th)},{0,0,1,0},{0,sin(th),0,-cos(th)}};
        double g = sqrt(1 + qh2 / stt), gb = qh / rstt;
        double invboost[4][4] = {{g,0,0,gb},{0,1,0,0},{0,0,1,0},{gb,0,0,g}};
        double q1[4] = {rs/2,0,0,rs/2}, q2[4] = {rs/2,0,0,-rs/2}, qH[4] = {Eh, qh*sin(th), 0, qh*cos(th)};
        transform(qt1, invboost); transform(qt2, invboost); transform(qt1, invrot); transform(qt2, invrot);
        double beta_tt = 2 * pt / rstt;
        double weight = (sqr(rs - mh) - 4 * mt * mt) * sqrt(stt * (stt - 4 * mt * mt) * kl) / (1024 * pow(M_PI, 4) * s * stt) * sin(th) * sin(tt);
        // OpenLoops: 5 numbers per particle (E, px, py, pz, m), process order parton, parton, t, tbar, H
        double pp[25] = {0}; const double* legs[5] = {q1, q2, qt1, qt2, qH}; const double ms[5] = {0, 0, mt, mt, mh};
        for (int i = 0; i < 5; i++) { for (int m = 0; m < 4; m++) pp[5 * i + m] = MT * legs[i][m]; pp[5 * i + 4] = MT * ms[i]; }
        ol_setparameter_double("mu", MT * rs / 2);
        double m2tree = 0, m2loop[3] = {0, 0, 0}, acc = 0, cc[10] = {0}, ewcc = 0;
        double ao2pi = alphas / (2 * M_PI);
        if (want_cc) {
            double tree2 = 0, loop2[3] = {0, 0, 0};
            ol_evaluate_loopcc(id, pp, &tree2, loop2, cc, &ewcc);
            fprintf(fout, "tree %.10e cc/tree:", tree2);
            for (int k = 0; k < 10; k++) fprintf(fout, " %.6f", cc[k] / tree2);
            fprintf(fout, "\n");
            continue;
        }
        ol_evaluate_loop(id, pp, &m2tree, m2loop, &acc);
        double tree2 = 0, loop2[3] = {0, 0, 0};
        ol_evaluate_loopcc(id, pp, &tree2, loop2, cc, &ewcc);
        double cc_tt = cc[gg ? 9 : 9];   /* set after inspecting the --cc output */
        fprintf(fout, "%.16e %.16e %.16e %.16e %.16e %.16e %.16e %.3e\n", weight, m2tree, m2loop[0] / ao2pi, m2loop[1] / ao2pi, m2loop[2] / ao2pi, cc_tt, beta_tt, acc);
        fflush(fout);
    }
    ol_finish();
    return 0;
}
