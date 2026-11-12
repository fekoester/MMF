//------------------------------------------------------------------------------------------------------------------//
// Master Memory Function (MMF): linear memory capacity of a delay-based reservoir computer with single-variable
// dynamics and small inputs (Koester, Yanchuk, Luedge, IEEE TNNLS 35, 7712 (2024), arXiv:2108.12643).
//
// Linearised and discretised over one virtual-node interval theta, the reservoir is the map
//     s_j = gamma I_j + p s_{j-1} + m s_{j-nu},   p = exp(a theta),  m = -(b/a)(1 - p),  nu = tau/theta,
// with I_j = u_{floor(j/N)} w_{j mod N}. Unrolling gives s_j = gamma sum_{i,k} C(i+k,i) p^i m^k I_{j-i-k nu}.
// Collecting the coefficients per recall step l gives the modified state matrix S~ (N x L) and
//     C_l = S~_l^T (S~ S~^T + lambda I)^+ S~_l,        MC = sum_l C_l.
//------------------------------------------------------------------------------------------------------------------//
#include <armadillo>
#include <boost/math/distributions/binomial.hpp>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

struct Par {
    double a = -0.5;
    double b = 0.4;
    double theta = 0.5;
    unsigned int N = 100;
    unsigned int tauInTheta = 141;
    double regularisationParameter = 1E-6;
    bool regularisationParameterRelative = true;
    double relativeThreshold = 1E-6;
    unsigned int maxIterations = 20000;  // largest total number of steps i + k considered
    unsigned int seed = 0;
    std::string maskFile;                 // optional: mask weights w_0..w_{N-1}, one per line
    std::string delayConvention = "paper";  // "paper": tau = nu theta; "legacy": v0.1 (tau = (nu + 1) theta)
    std::string filenameToWrite = "mmf_output.txt";
    std::vector<double> maskWeights;
};

static void printUsage() {
    std::cout <<
        "Usage: ./MMF [options]\n"
        "  -a A                               local linearisation (eigenvalue) a < 0 (default -0.5)\n"
        "  -b B                               delayed-feedback linearisation b (default 0.4)\n"
        "  -theta T                           virtual node distance theta (default 0.5)\n"
        "  -N N                               number of virtual nodes (default 100)\n"
        "  -tauInTheta NU                     delay tau = NU * theta (default 141)\n"
        "  -delayConvention paper|legacy      'legacy' reproduces v0.1, which modelled tau = (NU + 1) theta\n"
        "  -seed S                            seed of the random mask, uniform in [0, 1) (default 0)\n"
        "  -maskFile FILE                     use this mask instead (N values, w_n for node n)\n"
        "  -regularisationParameter L         Tikhonov parameter (default 1e-6)\n"
        "  -regularisationParameterRelative 0|1   L relative to max(S~ S~^T) (default 1)\n"
        "  -relativeThreshold E               drop terms below E (default 1e-6)\n"
        "  -maxIterations K                   largest i + k summed (default 20000)\n"
        "  -filename FILE                     output file (default mmf_output.txt)\n";
}

static std::string valueOf(int argc, char **argv, int &i) {
    if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value after ") + argv[i]);
    return std::string(argv[++i]);
}

static double numberOf(int argc, char **argv, int &i) {
    std::string flag = argv[i], v = valueOf(argc, argv, i);
    char *end = nullptr;
    double x = std::strtod(v.c_str(), &end);
    if (end == v.c_str() || *end != '\0') throw std::invalid_argument("not a number after " + flag + ": " + v);
    return x;
}

// Parse every flag first; the mask is drawn afterwards, so the result does not depend on the flag order.
Par parsArguments(int argc, char **argv) {
    Par par;
    for (int i = 1; i < argc; i++) {
        std::string f = argv[i];
        if (f == "-h" || f == "--help") { printUsage(); std::exit(0); }
        else if (f == "-a") par.a = numberOf(argc, argv, i);
        else if (f == "-b") par.b = numberOf(argc, argv, i);
        else if (f == "-theta") par.theta = numberOf(argc, argv, i);
        else if (f == "-N") par.N = (unsigned)numberOf(argc, argv, i);
        else if (f == "-tauInTheta") par.tauInTheta = (unsigned)numberOf(argc, argv, i);
        else if (f == "-relativeThreshold") par.relativeThreshold = numberOf(argc, argv, i);
        else if (f == "-regularisationParameter") par.regularisationParameter = numberOf(argc, argv, i);
        else if (f == "-regularisationParameterRelative") par.regularisationParameterRelative = numberOf(argc, argv, i) != 0;
        else if (f == "-maxIterations") par.maxIterations = (unsigned)numberOf(argc, argv, i);
        else if (f == "-seed") par.seed = (unsigned)numberOf(argc, argv, i);
        else if (f == "-maskFile") par.maskFile = valueOf(argc, argv, i);
        else if (f == "-delayConvention") par.delayConvention = valueOf(argc, argv, i);
        else if (f == "-filename") par.filenameToWrite = valueOf(argc, argv, i);
        else throw std::invalid_argument("unknown option " + f + " (see -h)");
    }
    if (par.a >= 0) throw std::invalid_argument("a must be negative (p = exp(a theta) < 1)");
    if (par.theta <= 0) throw std::invalid_argument("theta must be positive");
    if (par.N == 0) throw std::invalid_argument("N must be positive");
    if (par.tauInTheta == 0) throw std::invalid_argument("tauInTheta must be positive");
    if (par.delayConvention != "paper" && par.delayConvention != "legacy")
        throw std::invalid_argument("-delayConvention must be 'paper' or 'legacy'");

    if (!par.maskFile.empty()) {
        arma::vec w;
        if (!w.load(par.maskFile) || w.n_elem != par.N)
            throw std::runtime_error("could not read " + std::to_string(par.N) + " mask values from '" + par.maskFile + "'");
        par.maskWeights.assign(w.begin(), w.end());
    } else {
        std::mt19937 gen(par.seed);  // same mask as v0.1 for the same seed and N
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        par.maskWeights.resize(par.N);
        for (auto &w : par.maskWeights) w = dist(gen);
    }
    std::cout << "a=" << par.a << " b=" << par.b << " theta=" << par.theta << " N=" << par.N
              << " tauInTheta=" << par.tauInTheta << " (" << par.delayConvention << " convention)"
              << (par.maskFile.empty() ? " seed=" + std::to_string(par.seed) : " mask=" + par.maskFile) << std::endl;
    return par;
}

// C(i+k, i) p^i m^k, written via the binomial distribution: (-b/a)^k * Binom(i+k, 1-p)(k)
static double term(const Par &par, double p, unsigned int total, unsigned int k) {
    return std::pow(-par.b / par.a, (int)k) * boost::math::pdf(boost::math::binomial(total, 1.0 - p), k);
}

// Modified state matrix S~ (N x L): column l collects the coefficients of the input l clock cycles back.
arma::mat getModifiedState(const Par &par, double p) {
    const bool legacy = par.delayConvention == "legacy";
    const long N = par.N, nu = par.tauInTheta;
    std::vector<arma::vec> cols;
    auto column = [&](size_t l) -> arma::vec & {
        while (cols.size() <= l) cols.emplace_back(arma::vec(N, arma::fill::zeros));
        return cols[l];
    };
    bool converged = false;
    for (unsigned int total = 0; total < par.maxIterations && !converged; total++) {
        bool allSmall = true;
        for (unsigned int k = 0; k <= total; k++) {
            const double t = term(par, p, total, k);
            if (std::abs(t) >= par.relativeThreshold) allSmall = false;  // term(0,0) = 1, so this is relative
            if (legacy) {
                // v0.1: offset total + k*nu = i + k*(nu + 1); node order reversed (equivalent to a reversed mask)
                const long off = (long)total + (long)k * nu;
                for (long n = 0; n < N; n++) column((n + off) / N)(n) += t * par.maskWeights[(n + off) % N];
            } else {
                // node n of cycle c receives I_{cN + n - off}, off = i + k*nu: recall step ceil((off - n)/N),
                // mask index (n - off) mod N
                const long off = (long)(total - k) + (long)k * nu;
                for (long n = 0; n < N; n++) {
                    const long l = (off - n + N - 1) / N;  // ceil, off - n >= -(N-1)
                    const long idx = ((n - off) % N + N) % N;
                    if (l >= 0) column(l)(n) += t * par.maskWeights[idx];
                }
            }
        }
        if (allSmall) converged = true;
    }
    if (!converged)
        std::cerr << "warning: terms still above -relativeThreshold after -maxIterations " << par.maxIterations
                  << " (slow decay, a*theta close to 0?); the result is truncated" << std::endl;
    arma::mat S(N, cols.size());
    for (size_t l = 0; l < cols.size(); l++) S.col(l) = cols[l];
    return S;
}

arma::vec capacities(const Par &par, const arma::mat &S) {
    arma::mat G = S * S.t();
    double lambda = par.regularisationParameter;
    if (par.regularisationParameterRelative) lambda *= G.max();
    arma::mat Ginv = arma::pinv(G + lambda * arma::eye<arma::mat>(par.N, par.N));
    return arma::sum(S % (Ginv * S), 0).t();  // C_l = S_l^T Ginv S_l
}

int main(int argc, char **argv) {
    try {
        const auto start = std::chrono::steady_clock::now();
        Par par = parsArguments(argc, argv);
        arma::mat S = getModifiedState(par, std::exp(par.a * par.theta));
        arma::vec C = capacities(par, S);

        std::ofstream out(par.filenameToWrite);
        if (!out) throw std::runtime_error("cannot write '" + par.filenameToWrite + "'");
        out << "Steps\tCapacity\n" << std::setprecision(14);
        for (arma::uword l = 0; l < C.n_elem; l++) out << l << "\t" << C(l) << "\n";

        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        std::cout << std::setprecision(10) << "Total linear memory capacity: " << arma::accu(C) << " (N = " << par.N
                  << ", " << C.n_elem << " recall steps) -> " << par.filenameToWrite << std::endl;
        std::cout << "Elapsed time: " << (ms > 1000 ? std::to_string(ms / 1000) + " s" : std::to_string((int)ms) + " ms")
                  << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
