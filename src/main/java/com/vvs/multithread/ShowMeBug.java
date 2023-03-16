package com.vvs.multithread;

import java.util.concurrent.CountDownLatch;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 21:50 2022/3/8
 * @Modified By:
 */
public class ShowMeBug {

    private static volatile double balance;

    public synchronized void deposit(double money) {
        balance += money;
    }

    public double getBalance() {
        return balance;
    }

    public static void main(String[] args) {
        final CountDownLatch countDownLatch = new CountDownLatch(100);
        ShowMeBug account = new ShowMeBug();
        for (int i = 0;i < 100; i++) {
            Thread thread = new Thread(() -> {
                account.deposit(1);
                countDownLatch.countDown();
            });
            thread.start();
        }
        try {
            countDownLatch.await();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        System.out.println(account.getBalance());
    }
}
