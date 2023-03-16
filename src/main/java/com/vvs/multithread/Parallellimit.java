package com.vvs.multithread;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 22:22 2020/10/13
 * @Modified By:
 */
public class Parallellimit {

    public static void main(String[] args) {
        ExecutorService executorService = Executors.newSingleThreadExecutor();
        CountDownLatch countDownLatch = new CountDownLatch(100);
        for(int i = 0;i < 100; i++){

        }
    }
}

class CountRunnable implements Runnable{

    private CountDownLatch countDownLatch;

    public CountRunnable(CountDownLatch countDownLatch){
        this.countDownLatch = countDownLatch;
    }

    @Override
    public void run() {

    }
}
