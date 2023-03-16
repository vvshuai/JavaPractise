package com.vvs.codeforces;

import java.util.Scanner;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 23:37 2022/7/8
 * @Modified By:
 */
public class CF07081 {

    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        while (in.hasNext()) {
            int mod = (int) (1e9 + 7);
            String s = in.next();
            int n = s.length();
            long[] dp = new long[n + 1];
            dp[0] = 0;
            int last = -1;
            for (int i = 1 ;i <= n; i++) {
                if (s.charAt(i - 1) == 'a') {
                    dp[i] = dp[i - 1] + 1;
                    if (last != -1) {
                        dp[i] += dp[last];
                    }
                } else {
                    dp[i] = dp[i - 1];
                    if (s.charAt(i - 1) == 'b') {
                        last = i;
                    }
                }
                dp[i] %= mod;
            }
            System.out.println(dp[n]);
        }
    }
}
