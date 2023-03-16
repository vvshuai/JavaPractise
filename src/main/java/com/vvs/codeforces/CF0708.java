package com.vvs.codeforces;

import java.util.Scanner;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 0:04 2022/7/8
 * @Modified By:
 */
public class CF0708 {

    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        int mod = (int) (1e9 + 7);
        while (in.hasNext()) {
            int n = in.nextInt();
            int l = in.nextInt();
            int r = in.nextInt();
            int[] c = new int[3];
            int k = (r - l + 1) / 3;
            c[0] = k;
            c[1] = k;
            c[2] = k;
            for (int i = l + 3 * k;i <= r; i++) {
                if (i % 3 == 0) {
                    c[0]++;
                } else if (i % 3 == 1) {
                    c[1]++;
                } else {
                    c[2]++;
                }
            }
            long[][] dp = new long[n + 1][3];
            dp[1][0] = c[0];
            dp[1][1] = c[1];
            dp[1][2] = c[2];
            for (int i = 2;i <= n; i++) {
                dp[i][0] = (dp[i - 1][0] * c[0] + dp[i - 1][1] * c[2] + dp[i - 1][2] * c[1]) % mod;
                dp[i][1] = (dp[i - 1][0] * c[1] + dp[i - 1][1] * c[0] + dp[i - 1][2] * c[2]) % mod;
                dp[i][2] = (dp[i - 1][0] * c[2] + dp[i - 1][1] * c[1] + dp[i - 1][2] * c[0]) % mod;
            }
            System.out.println(dp[n][0]);
        }
    }
}
