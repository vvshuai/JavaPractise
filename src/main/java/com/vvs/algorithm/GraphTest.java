package com.vvs.algorithm;

import java.util.*;
import java.util.function.BiFunction;

public class GraphTest {
    long ans;
    public long minimumFuelCost(int[][] roads, int seats) {
        ans = 0;
        int n = roads.length + 1;
        List<Integer>[] graph = new List[n];
        for (int i = 0; i < n; i++) {
            graph[i] = new ArrayList<>();
        }
        for (int[] road : roads) {
            graph[road[0]].add(road[1]);
            graph[road[1]].add(road[0]);
        }
        BiFunction<Integer, Integer, Integer> function = new BiFunction<Integer, Integer, Integer>() {
            @Override
            public Integer apply(Integer x, Integer fa) {
                int ret = 1;
                for (int c : graph[x]) {
                    if (c != fa) {
                        ret += apply(c, x);
                    }
                }
                if (x != 0) {
                    ans += (ret + seats - 1) / seats;
                }
                return ret;
            }
        };
        function.apply(0, -1);
        return ans;
    }

    public int beautifulPartitions(String s, int k, int minLength) {
        int mod = (int) (1e9 + 7);
        char[] chars = s.toCharArray();
        int n = chars.length;
        if (k * minLength > n || isPrime(chars[0]) || !isPrime(chars[n - 1])) {
            return 0;
        }
        int[][] f = new int[k + 1][n + 1];
        f[0][0] = 1;
        for (int i = 1;i <= k; i++) {
            long tot = 0;
            for (int j = i * minLength; j + (k - i) * minLength <= n; j++) {
                if (canPartition(chars, j - minLength)) {
                    tot = (tot + f[i - 1][j - minLength]) % mod;
                }
                if (canPartition(chars, j)) f[i][j] = (int) tot;
            }
        }
        return f[k][n];
    }

    public boolean isPrime(char c) {
        return "2357".contains(c + "");
    }

    public boolean canPartition(char[] s, int j) {
        return j == 0 || j == s.length || (!isPrime(s[j - 1]) && isPrime(s[j]));
    }

    public static void main(String[] args) {
        long l = new GraphTest().minimumFuelCost(new int[][]{
                {3, 1}, {3, 2}, {1, 0}, {0, 4}, {0, 5}, {4, 6}
        }, 2);
        System.out.println(l);
    }
}
