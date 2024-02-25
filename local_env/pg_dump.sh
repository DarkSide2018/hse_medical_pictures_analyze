 docker exec d432ce88c998 pg_dump -U hse_medical -h localhost  > /home/roman/Documents/hse/dumpas/dump-11-01-2024.sql

cat /home/roman/Documents/hse/dumpas/dump-14-01-2024.sql | docker exec -i  fce4a81caa1a psql -U hse_medical -d hse_medical
