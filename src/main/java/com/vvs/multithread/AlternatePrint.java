package com.vvs.multithread;

import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 9:36 2020/9/8
 * @Modified By:
 */
public class AlternatePrint {

//    private static ReentrantLock lock = new ReentrantLock();
//    private static Condition condition1 = lock.newCondition();
//    private static Condition condition2 = lock.newCondition();
//
//    static class ThreadA extends Thread{
//
//        @Override
//        public void run() {
//            for(int i = 0;i < 10; i++){
//                lock.lock();
//                try{
//                    System.out.println(i+1);
//                    condition2.signal();
//                    condition1.await();
//                } catch (InterruptedException e) {
//                    e.printStackTrace();
//                } finally {
//                    lock.unlock();
//                }
//            }
//            condition1.signal();
//        }
//    }
//
//    static class ThreadB extends Thread{
//
//        @Override
//        public void run() {
//            for(int i = 0;i < 10; i++){
//                lock.lock();
//                try{
//                    char x = (char)('A' + i);
//                    System.out.println(x);
//                    condition1.signal();
//                    condition2.await();
//                } catch (InterruptedException e) {
//                    e.printStackTrace();
//                } finally {
//                    lock.unlock();
//                }
//            }
//            condition2.signal();
//        }
//    }

    public static void main(String[] args) {
        ReentrantLock lock = new ReentrantLock();
        Condition condition1 = lock.newCondition();
        Condition condition2 = lock.newCondition();
        Thread thread1 = new Thread(()->{
            try {
                lock.lock();
                for (int i=1;i<100;i+=2){
                    System.out.println("-----thread1-----" + i );
                    condition1.await();
                    condition2.signal();
                    Thread.sleep(500);
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            } finally {
                lock.unlock();
            }
        });
        Thread thread2 = new Thread(()->{
            try {
                lock.lock();
                for(int i=2;i<101;i+=2){
                    System.out.println("-----thread2-----" + i);
                    condition1.signal();
                    condition2.await();
                    Thread.sleep(500);
                }
            } catch (InterruptedException e){
                e.getStackTrace();
            } finally {
                lock.unlock();
            }

        });
        thread1.start();
        thread2.start();
    }
}
