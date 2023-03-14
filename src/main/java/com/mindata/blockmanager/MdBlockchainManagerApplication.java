package com.mindata.blockmanager;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

@SpringBootApplication
public class MdBlockchainManagerApplication {

	public static void main(String[] args) {

		Process pr;
		try {
			System.out.println("开始网络初始化");
			System.out.println("——————————————————————————————");
			String exe = "python3";
			String command = "D:\\研究生\\BFT算法改进\\md_blockchain_manager-master\\md_blockchain_manager-master\\src\\main\\python\\nodes_location.py";
			String[] cmdArr = new String[] { exe, command };
			Process process = Runtime.getRuntime().exec(cmdArr);
			BufferedReader in = new BufferedReader(new InputStreamReader(process.getInputStream(),"gb2312"));
			String line;
			while( ( line = in.readLine() ) != null ) {
				System.out.println(line);
			}
			in.close();
			process.waitFor();
//            int result = process.waitFor();
//            System.out.println("执行结果:" + result);
			System.out.println("——————————————————————————————");
			System.out.println("网络初始化完成");
		} catch (IOException | InterruptedException e) {
			e.printStackTrace();
		}
		SpringApplication.run(MdBlockchainManagerApplication.class, args);
	}
}
