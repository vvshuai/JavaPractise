package com.vvs.practise0608;

import com.google.common.hash.BloomFilter;
import com.google.common.hash.Funnels;

import java.util.concurrent.locks.AbstractQueuedSynchronizer;
import java.util.concurrent.locks.ReentrantLock;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 15:54 2020/9/3
 * @Modified By:
 */
public class BloomFilterTest {

    private static final int SIZE = 1000001;

    private static BloomFilter bloomFilter = BloomFilter.create(Funnels.integerFunnel(), SIZE);

    public static void main(String[] args) {
        for(int i = 0;i < SIZE; i++){
            bloomFilter.put(i);
        }
        long startTime = System.nanoTime();
    }

}
