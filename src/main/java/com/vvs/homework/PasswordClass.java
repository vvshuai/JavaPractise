package com.vvs.homework;

public class PasswordClass {

    public static void main(String[] args) {
        System.out.println(Rabin());
    }

    public static long Rabin() {
        return (32767 * (32767 + 1357)) % (199 * 211);
    }

    public static long ElGamal() {
        long m = 1;
        for (int i = 0;i < 100000; i++) {
            m *= 7;
            if (m % 31847 == 18074) {
                return i;
            }
        }
        return 0;
    }
}
