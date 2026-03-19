#include <iostream>
using namespace std;

// function to compute (base^exp) mod mod
long long modPow(long long base, long long exp, long long mod)
{
    long long result = 1;

    for(int i = 0; i < exp; i++)
        result = (result * base) % mod;

    return result;
}

int main()
{
    // public parameters
    int p = 23;
    int g = 5;

    // private keys
    int a = 6;
    int b = 15;

    cout << "Public values:\n";
    cout << "p = " << p << "  g = " << g << endl;

    // public keys
    int A = modPow(g, a, p);
    int B = modPow(g, b, p);

    cout << "\nUser1 Public Key = " << A << endl;
    cout << "User2 Public Key = " << B << endl;

    // shared secrets
    int secret1 = modPow(B, a, p);
    int secret2 = modPow(A, b, p);

    cout << "\nShared Secret at User1 = " << secret1 << endl;
    cout << "Shared Secret at User2 = " << secret2 << endl;

    if(secret1 == secret2)
        cout << "Both users obtained the same secret\n";

    /* ---------- Man in the Middle ---------- */

    int e1 = 7;
    int e2 = 9;

    int M1 = modPow(g, e1, p);
    int M2 = modPow(g, e2, p);

    cout << "\nAttacker Public Value to User2 = " << M1 << endl;
    cout << "Attacker Public Value to User1 = " << M2 << endl;

    int user1_secret = modPow(M2, a, p);
    int user2_secret = modPow(M1, b, p);

    cout << "\nUser1 Secret (with attacker) = " << user1_secret << endl;
    cout << "User2 Secret (with attacker) = " << user2_secret << endl;

    int attacker_user1 = modPow(A, e2, p);
    int attacker_user2 = modPow(B, e1, p);

    cout << "\nAttacker Secret with User1 = " << attacker_user1 << endl;
    cout << "Attacker Secret with User2 = " << attacker_user2 << endl;

    return 0;
}