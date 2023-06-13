FROM kalilinux/kali-rolling:latest

RUN apt update
RUN apt install sshpass -y

COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod 777 /usr/local/bin/dockerd-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
