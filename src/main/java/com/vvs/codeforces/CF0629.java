package com.vvs.codeforces;

import java.io.*;
import java.util.*;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 0:27 2022/6/30
 * @Modified By:
 */
public class CF0629 {

    public static void main(String[] args) throws IOException {
        StreamTokenizer in=new StreamTokenizer(new BufferedReader(new InputStreamReader(System.in)));
        while (in.nextToken() != StreamTokenizer.TT_EOF) {
            String s1 = in.sval;
            in.nextToken();
            String s2 = in.sval;
            if (s1.compareTo(s2) >= 0) {
                System.out.println("No such string");
            } else {
                int last = 0;
                StringBuilder sb = new StringBuilder();
                for (int i = s1.length() - 1;i >= 0; i--) {
                    char x = s1.charAt(i);
                    int cur = x - 'a';
                    cur += (i == s1.length() - 1 ? 1 : 0) + last;
                    if (cur >= 26) {
                        cur %= 26;
                        last = 1;
                    } else {
                        last = 0;
                    }
                    sb.insert(0, (char)('a' + cur));
                }
                if (sb.toString().compareTo(s2) >= 0) {
                    System.out.println("No such string");
                } else {
                    System.out.println(sb);
                }
            }
        }
    }

    public static int gcd(int a, int b) {
        return b == 0 ? a : gcd(b, a % b);
    }

}
