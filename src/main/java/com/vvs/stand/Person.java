package com.vvs.stand;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 20:43 2020/6/18
 * @Modified By:
 */
public class Person {

    private String username;
    private int userAge;
    private final int money  = 10000;

    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public int getUserAge() {
        return userAge;
    }

    public void setUserAge(int userAge) {
        this.userAge = userAge;
    }

    //借钱方法
    public int getMoney(){
        System.out.println("你借了 " + money + "元！");
        return money;
    }
    //还钱方法，单个参数
    public void repay(int money){
        System.out.println("你还了 " + money + "元！");
    }
    //还钱方法，多个参数
    public void repay(String userName,int money){
        System.out.println(userName+ " 还了 " + money + "元！");
    }
}
