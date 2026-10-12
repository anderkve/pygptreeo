// Tree-level ttH matrix element on the 5-variable phase space of arXiv:2412.09534.
// stdin: lines "beta2 frac_stt theta_h theta_t phi_t"; stdout: "weight me" per line,
// weight the phase-space density of the angular variables (m_t = 1 units), me the
// spin/colour-averaged |M|^2 from MadGraph (GeV^-2, overall constant irrelevant).
#include <cstdio>
#include <cmath>
#include <vector>
#include "CPPProcess.h"
static double sqr(double x) { return x * x; }
static double kallen(double a, double sb, double sc) { return (a - sqr(sb + sc)) * (a - sqr(sb - sc)); }
static void transform(double x[4], const double m[4][4]) {
    double y[4];
    for (int i = 0; i < 4; i++) y[i] = m[i][0]*x[0] + m[i][1]*x[1] + m[i][2]*x[2] + m[i][3]*x[3];
    for (int i = 0; i < 4; i++) x[i] = y[i];
}
int main(int argc, char** argv) {
    CPPProcess process;
    process.initProc(argc > 1 ? argv[1] : "../../Cards/param_card.dat");
    const double mt = process.getMasses()[2], mh = process.getMasses()[4];
    double beta2, frac, th, tt, ph;
    while (scanf("%lf %lf %lf %lf %lf", &beta2, &frac, &th, &tt, &ph) == 5) {
        double s = sqr(2*mt + mh) / (1 - beta2), sqrt_s = sqrt(s);
        double stt = 4*mt*mt + frac * (sqr(sqrt_s - mh) - 4*mt*mt), sqrt_stt = sqrt(stt);
        double pt = sqrt(stt - 4*mt*mt) / 2;
        double qt1[4] = {sqrt_stt/2, pt*sin(tt)*cos(ph), pt*sin(tt)*sin(ph), pt*cos(tt)};
        double qt2[4] = {qt1[0], -qt1[1], -qt1[2], -qt1[3]};
        double kl = kallen(stt, sqrt_s, mh), qh2 = kl / 4 / s, qh = sqrt(qh2), Eh = sqrt(qh2 + mh*mh);
        double invrot[4][4] = {{1,0,0,0},{0,-cos(th),0,-sin(th)},{0,0,1,0},{0,sin(th),0,-cos(th)}};
        double g = sqrt(1 + qh2/stt), gb = qh/sqrt_stt;
        double invboost[4][4] = {{g,0,0,gb},{0,1,0,0},{0,0,1,0},{gb,0,0,g}};
        double q1[4] = {sqrt_s/2, 0, 0, sqrt_s/2}, q2[4] = {sqrt_s/2, 0, 0, -sqrt_s/2};
        double qH[4] = {Eh, qh*sin(th), 0, qh*cos(th)};
        transform(qt1, invboost); transform(qt2, invboost); transform(qt1, invrot); transform(qt2, invrot);
        std::vector<double*> p = {q1, q2, qt1, qt2, qH};
        process.setMomenta(p);
        process.sigmaKin();
        double me = process.getMatrixElements()[0];
        // Phase-space weight in m_t = 1 units (dimensionless)
        double S = s/(mt*mt), STT = stt/(mt*mt), MH = mh/mt;
        double w = (sqr(sqrt(S) - MH) - 4) * sqrt(STT*(STT - 4) * kallen(STT, sqrt(S), MH)) / (1024*pow(M_PI, 4)*S*STT) * sin(th) * sin(tt);
        printf("%.16e %.16e\n", w, me);
    }
    return 0;
}
