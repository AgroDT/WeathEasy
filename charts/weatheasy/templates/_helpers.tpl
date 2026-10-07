{{- define "weatheasy.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- define "weatheasy.fullname" -}}
{{- default (printf "%s-%s" .Release.Name (include "weatheasy.name" .)) .Values.fullnameOverride | trunc 48 | trimSuffix "-" -}}
{{- end -}}
{{- define "weatheasy.selectorLabels" -}}
app.kubernetes.io/name: {{ include "weatheasy.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
{{- define "weatheasy.labels" -}}
{{ include "weatheasy.selectorLabels" . }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | quote }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}
{{- define "weatheasy.image" -}}
{{ printf "%s:%s" .Values.image.repository (default .Chart.AppVersion .Values.image.tag) | quote }}
{{- end -}}
{{- define "weatheasy.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "weatheasy.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}
{{- define "weatheasy.env" -}}
- name: HOME
  value: /var/lib/weatheasy
- name: PYTHONDONTWRITEBYTECODE
  value: "1"
{{- range $source, $settings := .Values.sources }}
{{- if $settings.store }}
- name: {{ printf "WEATHEASY__SOURCES__%s__STORE" (upper $source) }}
  value: {{ $settings.store | quote }}
{{- end }}
{{- end }}
{{- with .Values.s3.endpoint }}
- name: AWS_ENDPOINT_URL_S3
  value: {{ . | quote }}
{{- end }}
{{- with .Values.extraEnv }}
{{ toYaml . }}
{{- end }}
{{- end -}}
{{- define "weatheasy.volumeMounts" -}}
- name: work
  mountPath: /var/lib/weatheasy
- name: tmp
  mountPath: /tmp
{{- end -}}
{{- define "weatheasy.volumes" -}}
- name: work
  emptyDir:
    sizeLimit: {{ .Values.workVolume.sizeLimit }}
- name: tmp
  emptyDir:
    sizeLimit: {{ .Values.tmpVolume.sizeLimit }}
{{- end -}}
{{- define "weatheasy.podSettings" -}}
serviceAccountName: {{ include "weatheasy.serviceAccountName" . }}
automountServiceAccountToken: {{ .Values.serviceAccount.automountServiceAccountToken }}
securityContext:
  {{- toYaml .Values.podSecurityContext | nindent 2 }}
{{- with .Values.imagePullSecrets }}
imagePullSecrets:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- with .Values.nodeSelector }}
nodeSelector:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- with .Values.tolerations }}
tolerations:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- with .Values.affinity }}
affinity:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- end -}}
