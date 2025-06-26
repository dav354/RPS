import psutil

def get_system_stats():
    cpu = psutil.cpu_percent(interval=None)
    vm = psutil.virtual_memory()
    ram_str = f"{vm.used // (1024 * 1024)}MB / {vm.total // (1024 * 1024)}MB"

    cpu_temp = 0.0
    temps = psutil.sensors_temperatures()
    for label in ("cpu_thermal", "cpu-thermal", "coretemp"):
        if label in temps and temps[label]:
            cpu_temp = temps[label][0].current
            break

    return cpu, ram_str, round(cpu_temp, 1)
