 docker exec d432ce88c998 pg_dump -U hse_medical -h localhost  > /home/roman/Documents/hse/dumpas/dump-11-01-2024.sql

cat /home/roman/Documents/hse/dumpas/dump-8-01-2024.sql | docker exec -i cbf11dd2bd2d psql -U hse_medical -d hse_medical
