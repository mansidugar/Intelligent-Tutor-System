#include <iostream>
using namespace std;

// ─────────────────────────────────────────────
// Modular exponentiation: (base^exp) mod mod_val
// Using repeated squaring (no built-in power fn)
// ─────────────────────────────────────────────
long long mod_pow(long long base, long long exp, long long mod_val) {
    long long result = 1;
    base = base % mod_val;

    while (exp > 0) {
        // If exp is odd, multiply result by base
        if (exp % 2 == 1) {
            result = (result * base) % mod_val;
        }
        exp = exp / 2;          // Halve the exponent
        base = (base * base) % mod_val; // Square the base
    }
    return result;
}

// ─────────────────────────────────────────────
// Print a separator line
// ─────────────────────────────────────────────
void print_line() {
    for (int i = 0; i < 60; i++) std::cout << '-';
    std::cout << '\n';
}

// ─────────────────────────────────────────────
// Show step-by-step squaring method
// ─────────────────────────────────────────────
void show_steps(long long base, long long exp, long long mod_val) {
    std::cout << "  Computing " << base << "^" << exp << " mod " << mod_val << " step by step:\n";
    long long result = 1;
    long long b = base % mod_val;
    long long e = exp;

    std::cout << "  base=" << b << ", exp=" << e << "\n";
    while (e > 0) {
        if (e % 2 == 1) {
            long long prev = result;
            result = (result * b) % mod_val;
            std::cout << "  exp is odd → result = (" << prev << " * " << b
                      << ") mod " << mod_val << " = " << result << "\n";
        }
        e = e / 2;
        long long prev_b = b;
        b = (b * b) % mod_val;
        if (e > 0)
            std::cout << "  exp halved → base = (" << prev_b << "^2) mod "
                      << mod_val << " = " << b << ", exp = " << e << "\n";
    }
    std::cout << "  Result = " << result << "\n\n";
}

// ═════════════════════════════════════════════
// PART A: Standard DH Key Exchange
// ═════════════════════════════════════════════
void part_A() {
    std::cout << "\n";
    print_line();
    std::cout << "PART A: Diffie-Hellman Key Exchange\n";
    print_line();

    long long p = 23;  // Prime
    long long g = 5;   // Generator
    long long a = 6;   // User 1 private key
    long long b = 15;  // User 2 private key

    std::cout << "Public parameters: p = " << p
              << ", g = " << g << "\n";
    std::cout << "User 1 private key: a = " << a << "\n";
    std::cout << "User 2 private key: b = " << b << "\n\n";

    // Step 1: User 1 public key A = g^a mod p
    std::cout << "STEP 1: User 1 Public Key  A = g^a mod p = "
              << g << "^" << a << " mod " << p << "\n";
    show_steps(g, a, p);
    long long A = mod_pow(g, a, p);
    std::cout << "  >>> User 1 Public Key A = " << A << "\n\n";

    // Step 2: User 2 public key B = g^b mod p
    std::cout << "STEP 2: User 2 Public Key  B = g^b mod p = "
              << g << "^" << b << " mod " << p << "\n";
    show_steps(g, b, p);
    long long B = mod_pow(g, b, p);
    std::cout << "  >>> User 2 Public Key B = " << B << "\n\n";

    // Step 3: Shared secret at User 1's side: S = B^a mod p
    std::cout << "STEP 3: Shared Secret (User 1 side)  S = B^a mod p = "
              << B << "^" << a << " mod " << p << "\n";
    show_steps(B, a, p);
    long long S1 = mod_pow(B, a, p);
    std::cout << "  >>> User 1's Shared Secret = " << S1 << "\n\n";

    // Step 4: Shared secret at User 2's side: S = A^b mod p
    std::cout << "STEP 4: Shared Secret (User 2 side)  S = A^b mod p = "
              << A << "^" << b << " mod " << p << "\n";
    show_steps(A, b, p);
    long long S2 = mod_pow(A, b, p);
    std::cout << "  >>> User 2's Shared Secret = " << S2 << "\n\n";

    // Step 5: Verification
    std::cout << "STEP 5: Verification\n";
    std::cout << "  User 1 secret: " << S1 << "\n";
    std::cout << "  User 2 secret: " << S2 << "\n";
    if (S1 == S2)
        std::cout << "  RESULT: Both secrets MATCH! Shared key = "
                  << S1 << " [SUCCESS]\n";
    else
        std::cout << "  RESULT: Secrets do NOT match! [FAILURE]\n";
}
