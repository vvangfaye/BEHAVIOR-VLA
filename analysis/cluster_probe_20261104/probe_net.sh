echo NET_BEGIN; date
for u in https://huggingface.co https://pypi.org/simple/ https://github.com https://files.pythonhosted.org https://raw.githubusercontent.com; do
  printf "%s " "$u"; curl -s -o /dev/null -m 15 -w "%{http_code} %{time_total}s\n" "$u" || echo "fail"; done
env | grep -i -E "^(http|https|no)_proxy=" | sed -E 's#//[^@]*@#//<cred>@#' 
echo "HF 1GB download test:"; t0=$(date +%s); curl -s -L -m 240 -o /tmp/hf_test.bin -r 0-1073741823 https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT/resolve/main/model.safetensors; echo "rc=$? bytes=$(stat -c %s /tmp/hf_test.bin 2>/dev/null) secs=$(( $(date +%s) - t0 ))"
which uv pip3 git 2>&1 | head -3; whoami; echo "HOME=$HOME"
echo NET_END
