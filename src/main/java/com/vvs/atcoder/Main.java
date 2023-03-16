package com.vvs.atcoder;

import java.io.*;
import java.math.BigInteger;
import java.util.*;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 22:31 2021/12/20
 * @Modified By:
 */
public class Main {

    public static void main(String[] args) throws IOException {

        Scanner in = new Scanner(System.in);
        while (in.hasNext()) {
            int t = in.nextInt();
            while (t-- > 0) {
                String s = in.next();
                String c = in.next();
                int[] cur = new int[26];
                Arrays.fill(cur, -1);

                for (int i = s.length() - 1;i >= 0; i--) {
                    int vv = s.charAt(i) - 'A';
                    if (cur[vv] == -1) {
                        cur[vv] = i;
                    }
                }
                int k = 0;
                StringBuilder sb = new StringBuilder(s);
                for (int i = 0;i < sb.length(); i++) {
                    int vv = sb.charAt(i) - 'A';
                    while (k < vv && cur[k] < i) {
                        k++;
                    }
                    if (k < vv) {
                        char x = sb.charAt(i);
                        sb.setCharAt(i, sb.charAt(cur[k]));
                        sb.setCharAt(cur[k], x);
                        break;
                    }
                }
                if (sb.toString().compareTo(c) < 0) {
                    System.out.println(sb);
                } else {
                    System.out.println("---");
                }
            }
        }
    }

}
