# The AWS Emulator, as AWS documents it for Docker
# (https://docs.aws.amazon.com/emulator/latest/userguide/getting-started-docker.html).
# AWS publishes no prebuilt image: this installs the Linux package.
#
# One change: the credentials folder is created in the image, owned by the emulator's user,
# so a named volume mounted there starts writable and other containers can read the
# credentials, config and CA certificate the emulator writes into it.
FROM public.ecr.aws/amazonlinux/amazonlinux:2023-minimal
ARG PACKAGE=https://downloads.aws.com/aws-emulator/packages/aws-emulator/LTS/latest/aws-emulator-x86_64.rpm
ADD ${PACKAGE} /tmp/aws-emulator.rpm
RUN rpm -i /tmp/aws-emulator.rpm && rm /tmp/aws-emulator.rpm \
    && mkdir -p /home/emulator/.aws/emulator && chown -R 1000 /home/emulator
USER 1000
ENV HOME=/home/emulator
EXPOSE 9297
ENTRYPOINT ["aws-emulator", "-b", "0.0.0.0"]
