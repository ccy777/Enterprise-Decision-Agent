#!/bin/sh
set -eu

readonly_user=$(printf '%s' "${MYSQL_APP_READONLY_USER}" | sed "s/'/''/g")
readonly_password=$(printf '%s' "${MYSQL_APP_READONLY_PASSWORD}" | sed "s/'/''/g")

mysql -uroot -p"${MYSQL_ROOT_PASSWORD}" <<SQL
CREATE USER IF NOT EXISTS '${readonly_user}'@'%' IDENTIFIED BY '${readonly_password}';
GRANT SELECT ON enterprise_operations.* TO '${readonly_user}'@'%';
FLUSH PRIVILEGES;
SQL
