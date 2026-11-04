# Jenkins LTS con lo necesario para el Jenkinsfile de CDProd:
# - Python 3.13 (vía uv) para instalar requirements y correr pytest/ruff/métricas.
# - Docker CLI para construir la imagen de la API usando el Docker del host.
# - Plugins y configuración como código (JCasC): job, seguridad y SMTP listos al iniciar.
FROM jenkins/jenkins:lts-jdk17

USER root

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /usr/local/bin/
COPY --from=docker:28-cli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=docker:28-cli /usr/local/libexec/docker/cli-plugins/docker-buildx /usr/local/libexec/docker/cli-plugins/docker-buildx

ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_CACHE_DIR=/var/jenkins_home/.cache/uv \
    UV_LINK_MODE=copy
RUN uv python install 3.13 \
    && git config --system --add safe.directory '*'

COPY plugins.txt /usr/share/jenkins/ref/plugins.txt
RUN jenkins-plugin-cli --plugin-file /usr/share/jenkins/ref/plugins.txt

COPY casc.yaml /var/jenkins_config/casc.yaml
ENV CASC_JENKINS_CONFIG=/var/jenkins_config/casc.yaml \
    JAVA_OPTS="-Djenkins.install.runSetupWizard=false -Dhudson.plugins.git.GitSCM.ALLOW_LOCAL_CHECKOUT=true"
