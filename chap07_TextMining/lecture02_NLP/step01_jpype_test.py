# -*- coding: utf-8 -*-
"""
step01_jpype_test

JAVA 가상머신 사용을 위한 패키지 설치와 테스트

"""
import jpype



# 1. 
path = jpype.getDefaultJVMPath()
path # 'C:\\Program Files\\Java\\jre1.8.0_151\\bin\\server\\jvm.dll'
