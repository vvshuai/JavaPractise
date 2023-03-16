package com.vvs.stand;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 20:45 2020/6/18
 * @Modified By:
 */
public class Reflection {

    public static void main(String[] args) throws ClassNotFoundException, IllegalAccessException, InstantiationException,
            NoSuchMethodException, InvocationTargetException {
        Class<?> userClass = Class.forName("com.vvs.stand.Person");
        Person person = (Person)userClass.newInstance();

        System.out.println("第一次借钱:");
        int money = person.getMoney();
        System.out.println("实际拿到钱为:" + money);
        System.out.println("-------分割线----------");

        System.out.println("第二次借钱:");
        Method getMoney = userClass.getMethod("getMoney");
        Object money2 = getMoney.invoke(person);
        System.out.println("实际拿到钱为:" + money);
        System.out.println("-------分割线----------");

        System.out.println("第一次还钱:");
        Method repay1 = userClass.getMethod("repay", int.class);
        repay1.invoke(person, 3000);
        System.out.println("--------分割线---------");

        System.out.println("第二次还钱:");
        Method repay2 = userClass.getMethod("repay", String.class, int.class);
        repay2.invoke(person, "vvs", 5000);
    }
}
