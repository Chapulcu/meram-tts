#!/bin/bash
# Eski adı korumak için: API ve Streamlit uygulaması artık birlikte yönetiliyor. Ayrıntılar: servis/servisler.sh
exec bash "$(dirname "$0")/servisler.sh" "$@"
