package com.vvs.practise0608;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 17:06 2020/11/3
 * @Modified By:
 */
public interface Action {

    void eat();

    /** 抽象类中可以有自己的实现 */
    default void run(){
        System.out.println("run..");
    }
}

class Dog implements Action{

    @Override
    public void eat() {
        System.out.println("eat beef");
    }

    public static void main(String[] args) {
        Dog dog = new Dog();
        dog.eat();
        dog.run();
    }
}

