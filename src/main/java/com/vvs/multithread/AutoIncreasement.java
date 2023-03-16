package com.vvs.multithread;

import com.vvs.jvm0609.T;

import java.util.concurrent.CountDownLatch;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 22:29 2020/10/13
 * @Modified By:
 */
public class AutoIncreasement {

    private static volatile int count = 0;

    private static final CountDownLatch countDownLatch = new CountDownLatch(5);

    public static void main(String[] args) {
        TaskRunnale taskRunnale1 = new TaskRunnale();
        TaskRunnale taskRunnale2 = new TaskRunnale();
        TaskRunnale taskRunnale3 = new TaskRunnale();
        TaskRunnale taskRunnale4 = new TaskRunnale();
        TaskRunnale taskRunnale5 = new TaskRunnale();
        new Thread(taskRunnale1).start();
        new Thread(taskRunnale2).start();
        new Thread(taskRunnale3).start();
        new Thread(taskRunnale4).start();
        new Thread(taskRunnale5).start();

        try {
            countDownLatch.await();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        System.out.println(count);
    }

    static class TaskRunnale implements Runnable{

        @Override
        public void run() {
            synchronized (TaskRunnale.class){
                for(int i = 0;i < 10; i++){
                    count++;
                }
                countDownLatch.countDown();
            }
        }
    }
}

