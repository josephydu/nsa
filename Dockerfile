FROM mirrors.tencent.com/nvidia-infer/nvidia-infer:thunderkittens

# 设置环境变量
ENV http_proxy=http://star-proxy.oa.com:3128
ENV https_proxy=http://star-proxy.oa.com:3128
ENV no_proxy=docker.yard.oa.com,geminihub.oa.com,mirrors.cloud.tencent.com,mirrors.tencent.com,hub.oa.com,docker.oa.com,csighub.tencentyun.com,bk.artifactory.oa.com,10.0.0.0/8,100.64.0.0/10,9.0.0.0/8

RUN apt-get update && apt-get install -y git && apt-get clean

RUN git clone https://github.com/josephydu/sglang.git /sglang

WORKDIR /sglang

RUN git checkout wepoints-sglang-go

RUN pip install sgl-kernel --force-reinstall --no-deps
RUN pip install -e "python[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer/

# 清理代理环境变量（可选）
ENV http_proxy=""
ENV https_proxy=""

