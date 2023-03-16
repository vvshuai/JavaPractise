package com.vvs.multithread;

import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 11:07 2020/9/1
 * @Modified By:
 */
public class Repeated {

    private static ReentrantLock lock = new ReentrantLock();
    private static Condition A = lock.newCondition();
    private static Condition B = lock.newCondition();
    private static Condition C = lock.newCondition();
    private static int count = 0;
    private final static int SIZE = 10;
    private final static int MAX = 3;

    static class ThreadA extends Thread{

        @Override
        public void run() {
            lock.lock();
            try{
                for(int i = 0;i < SIZE; i++){
                    while(count % MAX != 0){
                        A.await();
                    }
                    System.out.print("A");
                    count++;
                    B.signal();
                }
            }catch (Exception e){
                e.printStackTrace();
            }finally{
                lock.unlock();
            }
        }
    }

    static class ThreadB extends Thread{
        @Override
        public void run() {
            lock.lock();
            try {
                for(int i = 0;i < SIZE; i++){
                    while(count % MAX != 1){
                        B.await();
                    }
                    System.out.print("B");
                    count++;
                    C.signal();
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            } finally {
                lock.unlock();
            }
        }
    }

    static class ThreadC extends Thread{
        @Override
        public void run() {
            lock.lock();
            try{
                for(int i = 0;i < SIZE; i++){
                    while(count % MAX != 2){
                        C.await();
                    }
                    System.out.print("C");
                    count++;
                    A.signal();
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }finally {
                lock.unlock();
            }
        }
    }

    public static void main(String[] args) {
        new ThreadA().start();
        new ThreadB().start();
        new ThreadC().start();
    }
}
