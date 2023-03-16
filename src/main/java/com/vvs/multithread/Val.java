package com.vvs.multithread;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 11:54 2020/9/3
 * @Modified By:
 */
public class Val<T> {

    T v;

    public void set(T _v){
        v = _v;
    }

    public T get(){
        return v;
    }
}
