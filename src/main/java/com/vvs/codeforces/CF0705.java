package com.vvs.codeforces;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.StreamTokenizer;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 23:59 2022/7/5
 * @Modified By:
 */
public class CF0705 {

    public static void main(String[] args) throws IOException {
        int mod = 1000000007;
        StreamTokenizer in=new StreamTokenizer(new BufferedReader(new InputStreamReader(System.in)));
        while (in.nextToken() != StreamTokenizer.TT_EOF) {
            int n = (int) in.nval;
            // 走i条棱到顶点的方案数
            long cur1 = 1;
            // 走i条棱不在顶点的方案数
            long cur2 = 0;
            for (int i = 1;i <= n; i++) {
                long newCur1 = cur2 * 3;
                long newCur2 = cur1 + cur2 * 2;
                cur1 = newCur1 % mod;
                cur2 = newCur2 % mod;
            }
            System.out.println(cur1);
        }
    }
}
