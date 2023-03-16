package com.vvs;

import com.vvs.shape.JoinDemo1;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;
import org.mockito.junit.MockitoJUnitRunner;

/**
 * @Author: vvshuai
 * @Description:
 * @Date: Created in 14:11 2020/8/1
 * @Modified By:
 */
public class Leetcode {

    @Mock
    JoinDemo1 joinDemo1;

    @Test
    public void testMock() {
        MockitoAnnotations.initMocks(this);
        Mockito.when(joinDemo1.hello("vvs")).thenReturn(1);
        Assert.assertEquals(1, joinDemo1.hello("vvs"));
    }
}
