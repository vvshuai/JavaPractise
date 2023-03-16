package com.vvs.algorithm;

import java.util.Arrays;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 0:23 2022/2/20
 * @Modified By:
 */
public class BITree {

    private int[] f = new int[100010];

    public void inc(int x) {
        for (;x <= 100000; x += x & (-x)) {
            f[x]++;
        }
    }

    public int calc(int x) {
        int res = 0;
        for (; x > 0; x -= x & (-x)) {
            res += f[x];
        }
        return res;
    }

    public void clear() {
        Arrays.fill(f, 0);
    }
}
