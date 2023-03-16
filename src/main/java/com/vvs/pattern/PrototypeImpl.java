package com.vvs.pattern;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 14:54 2020/7/1
 * @Modified By:
 */
public class PrototypeImpl extends AbstarctPrototype{

    private String filed;

    public PrototypeImpl(String filed){
        this.filed = filed;
    }

    @Override
    AbstarctPrototype myClone() {
        return new PrototypeImpl(filed);
    }

    @Override
    public String toString(){
        return filed;
    }

}
