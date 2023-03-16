package com.vvs.practise0608;

import java.util.concurrent.locks.*;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 21:09 2020/6/8
 * @Modified By:
 */
public class Thread0608 {

    static Lock lock = new ReentrantLock();
    static Condition conditiont1 = lock.newCondition();
    static Condition conditiont2 = lock.newCondition();

    public static void main(String[] args){
        char[] N = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".toCharArray();
        int[] C = new int[26];

        for(int i = 0;i < 26; i++){
            C[i] = i+1;
        }
        new Thread(()->{
            lock.lock();

            try{
                for(char c : N){
                    System.out.print(c);
                    conditiont1.await();
                    conditiont2.signal();
                }
            }catch (Exception e){
                e.printStackTrace();
            }finally {
                lock.unlock();
            }


        },"t1").start();
        new Thread(()->{
            lock.lock();
            try{
                for(int x : C){
                    System.out.print(x);
                    conditiont1.signal();
                    conditiont2.await();
                }
            }catch (Exception e){
                e.printStackTrace();
            }finally {
                lock.unlock();
            }

        },"t2").start();
    }
}
