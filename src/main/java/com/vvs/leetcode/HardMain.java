package com.vvs.leetcode;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 18:16 2022/7/10
 * @Modified By:
 */
public class HardMain {

    public static void main(String[] args) {
        for (int i = 0;i < 10; i++) {
            new Thread(() -> System.out.println(UUID.randomUUID())).start();
        }
    }
}
