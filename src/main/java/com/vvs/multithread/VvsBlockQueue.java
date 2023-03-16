package com.vvs.multithread;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 13:46 2020/7/15
 * @Modified By:
 */
public class VvsBlockQueue {
    private List<Integer> container = new ArrayList<>();
    private volatile int size;
    private volatile int capacity;
    private Lock lock = new ReentrantLock();
    /**
     * @Description: isNULL判断是否为空， isFull队列是否满
     */
    private final Condition isNUll = lock.newCondition();
    private final Condition isFull = lock.newCondition();

    VvsBlockQueue(int capacity){
        this.capacity = capacity;
    }

    /**
     * @Description: 添加方法
     * @return: void
     */
    public void add(int data){
        try{
            lock.lock();
            try{
                while(size >= capacity){
                    System.out.println("阻塞队列满了");
                    isFull.await();
                }
            }catch (Exception e){
                isFull.signal();
                e.printStackTrace();
            }
            ++size;
            container.add(data);
            isNUll.signal();
        }finally {
            lock.unlock();
        }
    }
}
