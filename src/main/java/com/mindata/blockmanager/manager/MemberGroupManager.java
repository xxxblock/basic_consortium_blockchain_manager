package com.mindata.blockmanager.manager;

import com.mindata.blockmanager.repository.MemberGroupRepository;
import org.springframework.stereotype.Component;

import javax.annotation.Resource;

@Component
public class MemberGroupManager {
    @Resource
    private MemberGroupRepository memberGroupRepository;
}
