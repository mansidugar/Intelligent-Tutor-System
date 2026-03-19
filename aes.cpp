#include <iostream>
#include <vector>
#include <iomanip>
#include <cstring>

using namespace std;

int sbox[256], inv_sbox[256];
int rcon[] = {0,1,2,4,8,16,32,64,128,27,54};

/* -------- Galois Field Multiplication -------- */
int gfMul(int a, int b)
{
    int res = 0;

    while(b)
    {
        if(b & 1) res ^= a;

        bool high = a & 0x80;
        a <<= 1;

        if(high) a ^= 0x1B;
        a &= 0xFF;

        b >>= 1;
    }
    return res;
}

/* -------- Initialize S-Box -------- */
void initSBox()
{
    for(int i=0;i<256;i++)
    {
        int val=i, inv=0;

        if(i!=0)
        {
            for(int j=0;j<6;j++)
                val = gfMul(gfMul(val,val), i);

            inv = gfMul(val,val);
        }

        int x = inv;

        int y = (x ^ (x<<1 | x>>7) ^
                    (x<<2 | x>>6) ^
                    (x<<3 | x>>5) ^
                    (x<<4 | x>>4) ^
                    0x63) & 255;

        sbox[i] = y;
        inv_sbox[y] = i;
    }
}

/* -------- Key Expansion -------- */
void expandKey(int key[16], int roundKey[176])
{
    for(int i=0;i<16;i++)
        roundKey[i] = key[i];

    for(int i=4;i<44;i++)
    {
        int temp[4];

        for(int j=0;j<4;j++)
            temp[j] = roundKey[(i-1)*4 + j];

        if(i % 4 == 0)
        {
            int t = temp[0];

            temp[0] = sbox[temp[1]];
            temp[1] = sbox[temp[2]];
            temp[2] = sbox[temp[3]];
            temp[3] = sbox[t];

            temp[0] ^= rcon[i/4];
        }

        for(int j=0;j<4;j++)
            roundKey[i*4+j] = roundKey[(i-4)*4+j] ^ temp[j];
    }
}

/* -------- Mix Columns -------- */
void mixColumn(int *state,int a,int b,int c,int d)
{
    for(int i=0;i<4;i++)
    {
        int *p = state + i*4;

        int w=p[0], x=p[1], y=p[2], z=p[3];

        p[0] = gfMul(a,w)^gfMul(b,x)^gfMul(c,y)^gfMul(d,z);
        p[1] = gfMul(d,w)^gfMul(a,x)^gfMul(b,y)^gfMul(c,z);
        p[2] = gfMul(c,w)^gfMul(d,x)^gfMul(a,y)^gfMul(b,z);
        p[3] = gfMul(b,w)^gfMul(c,x)^gfMul(d,y)^gfMul(a,z);
    }
}

/* -------- Permutation -------- */
void permute(int *state,int *perm)
{
    int temp[16];

    for(int i=0;i<16;i++)
        temp[i] = state[perm[i]];

    for(int i=0;i<16;i++)
        state[i] = temp[i];
}

/* -------- AES Encryption Block -------- */
void encryptBlock(int state[16],int roundKey[176])
{
    int shiftRow[] =
    {0,5,10,15,4,9,14,3,8,13,2,7,12,1,6,11};

    for(int i=0;i<16;i++)
        state[i] ^= roundKey[i];

    for(int round=1; round<=10; round++)
    {
        for(int i=0;i<16;i++)
            state[i] = sbox[state[i]];

        permute(state,shiftRow);

        if(round<10)
            mixColumn(state,2,3,1,1);

        for(int i=0;i<16;i++)
            state[i] ^= roundKey[round*16+i];
    }
}

/* -------- AES Decryption Block -------- */
void decryptBlock(int state[16],int roundKey[176])
{
    int invShift[] =
    {0,13,10,7,4,1,14,11,8,5,2,15,12,9,6,3};

    for(int i=0;i<16;i++)
        state[i] ^= roundKey[160+i];

    for(int round=9; round>=0; round--)
    {
        permute(state,invShift);

        for(int i=0;i<16;i++)
            state[i] = inv_sbox[state[i]];

        for(int i=0;i<16;i++)
            state[i] ^= roundKey[round*16+i];

        if(round>0)
            mixColumn(state,14,11,13,9);
    }
}

/* -------- CBC Encryption -------- */
vector<int> encryptCBC(string text,int key[16],int iv[16])
{
    int roundKey[176];
    expandKey(key,roundKey);

    vector<int> data(text.begin(),text.end());
    vector<int> cipher;

    int pad = 16 - data.size()%16;
    data.insert(data.end(),pad,pad);

    int prev[16];
    for(int i=0;i<16;i++) prev[i]=iv[i];

    for(int i=0;i<data.size();i+=16)
    {
        int block[16];

        for(int j=0;j<16;j++)
            block[j] = data[i+j] ^ prev[j];

        encryptBlock(block,roundKey);

        for(int j=0;j<16;j++)
        {
            cipher.push_back(block[j]);
            prev[j] = block[j];
        }
    }
    return cipher;
}

/* -------- CBC Decryption -------- */
string decryptCBC(vector<int> cipher,int key[16],int iv[16])
{
    int roundKey[176];
    expandKey(key,roundKey);

    string result;

    int prev[16];
    for(int i=0;i<16;i++) prev[i]=iv[i];

    for(int i=0;i<cipher.size();i+=16)
    {
        int block[16], saved[16];

        for(int j=0;j<16;j++)
        {
            block[j] = cipher[i+j];
            saved[j] = cipher[i+j];
        }

        decryptBlock(block,roundKey);

        for(int j=0;j<16;j++)
        {
            result += char(block[j]^prev[j]);
            prev[j] = saved[j];
        }
    }

    return result.substr(0,result.size()-result.back());
}

/* -------- MAIN -------- */
int main()
{
    initSBox();

    string plaintext,keyHex;

    cout<<"Enter plaintext : ";
    getline(cin,plaintext);

    cout<<"Enter key (32 hex chars): ";
    cin>>keyHex;

    int key[16];
    int iv[16]={0};

    for(int i=0;i<16;i++)
        key[i] = stoi(keyHex.substr(i*2,2),nullptr,16);

    vector<int> cipher = encryptCBC(plaintext,key,iv);
    string decrypted = decryptCBC(cipher,key,iv);

    cout<<"\nCiphertext : ";

    for(int v:cipher)
        cout<<hex<<setw(2)<<setfill('0')<<v;

    cout<<"\nDecrypted  : "<<decrypted<<endl;

    cout<<"Status     : "
        <<(decrypted==plaintext ?
           "Decryption Successful":"FAILED")
        <<endl;
}