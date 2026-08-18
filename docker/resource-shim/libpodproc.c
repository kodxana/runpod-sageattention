#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/*
 * A deliberately narrow /proc/meminfo compatibility shim.
 *
 * The library is loaded only by the free/htop/top wrappers installed beside
 * it.  It never belongs in /etc/ld.so.preload or a global LD_PRELOAD.  When a
 * finite memory cgroup cannot be resolved with confidence, every hook opens
 * the real /proc/meminfo unchanged (fail-open).
 */

#include <ctype.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef MFD_CLOEXEC
#define MFD_CLOEXEC 0x0001U
#endif

#define READ_BUFFER_SIZE (256U * 1024U)
#define UNLIMITED_THRESHOLD (1ULL << 60)

typedef FILE *(*fopen_function)(const char *, const char *);
typedef int (*open_function)(const char *, int, ...);
typedef int (*openat_function)(int, const char *, int, ...);
typedef int (*fortified_open_function)(const char *, int);
typedef int (*fortified_openat_function)(int, const char *, int);

static fopen_function real_fopen;
static fopen_function real_fopen64;
static open_function real_open;
static open_function real_open64;
static openat_function real_openat;
static openat_function real_openat64;
static fortified_open_function real___open_2;
static fortified_open_function real___open64_2;
static fortified_openat_function real___openat_2;
static fortified_openat_function real___openat64_2;
static __thread int hook_active;

struct cgroup_location {
    int version;
    char membership[PATH_MAX];
    char mount_root[PATH_MAX];
    char mount_point[PATH_MAX];
    char current[PATH_MAX];
};

struct memory_snapshot {
    unsigned long long host_total;
    unsigned long long limit;
    unsigned long long current;
    unsigned long long inactive_file;
    unsigned long long swap_limit;
    unsigned long long swap_current;
};

static void resolve_symbols(void) {
    if (!real_fopen)
        real_fopen = (fopen_function)dlsym(RTLD_NEXT, "fopen");
    if (!real_fopen64)
        real_fopen64 = (fopen_function)dlsym(RTLD_NEXT, "fopen64");
    if (!real_open)
        real_open = (open_function)dlsym(RTLD_NEXT, "open");
    if (!real_open64)
        real_open64 = (open_function)dlsym(RTLD_NEXT, "open64");
    if (!real_openat)
        real_openat = (openat_function)dlsym(RTLD_NEXT, "openat");
    if (!real_openat64)
        real_openat64 = (openat_function)dlsym(RTLD_NEXT, "openat64");
    if (!real___open_2)
        real___open_2 =
            (fortified_open_function)dlsym(RTLD_NEXT, "__open_2");
    if (!real___open64_2)
        real___open64_2 =
            (fortified_open_function)dlsym(RTLD_NEXT, "__open64_2");
    if (!real___openat_2)
        real___openat_2 =
            (fortified_openat_function)dlsym(RTLD_NEXT, "__openat_2");
    if (!real___openat64_2)
        real___openat64_2 =
            (fortified_openat_function)dlsym(RTLD_NEXT, "__openat64_2");
}

static bool disabled(void) {
    const char *value = getenv("PODPROC_DISABLE");
    if (!value || !*value)
        return false;
    return strcmp(value, "0") != 0 && strcasecmp(value, "false") != 0 &&
           strcasecmp(value, "no") != 0 && strcasecmp(value, "off") != 0;
}

static bool is_meminfo_path(const char *path) {
    return path && strcmp(path, "/proc/meminfo") == 0;
}

static ssize_t raw_read_file(const char *path, char *buffer, size_t capacity) {
    if (!buffer || capacity < 2)
        return -1;
    int fd = (int)syscall(SYS_openat, AT_FDCWD, path, O_RDONLY | O_CLOEXEC, 0);
    if (fd < 0)
        return -1;

    size_t used = 0;
    while (used + 1 < capacity) {
        ssize_t count = read(fd, buffer + used, capacity - used - 1);
        if (count > 0) {
            used += (size_t)count;
            continue;
        }
        if (count < 0 && errno == EINTR)
            continue;
        if (count < 0) {
            close(fd);
            return -1;
        }
        break;
    }
    close(fd);
    buffer[used] = '\0';
    return (ssize_t)used;
}

static bool raw_file_exists(const char *path) {
    char byte[2];
    return raw_read_file(path, byte, sizeof(byte)) >= 0;
}

static bool join_path(char *output, size_t size, const char *directory,
                      const char *name) {
    if (!output || !size || !directory || !name)
        return false;
    if (!name[0]) {
        int count = snprintf(output, size, "%s", directory);
        return count >= 0 && (size_t)count < size;
    }
    int count = snprintf(output, size, "%s%s%s", directory,
                         directory[0] && directory[strlen(directory) - 1] == '/'
                             ? ""
                             : "/",
                         name[0] == '/' ? name + 1 : name);
    return count >= 0 && (size_t)count < size;
}

static bool csv_contains(const char *csv, const char *value) {
    if (!csv || !value)
        return false;
    size_t wanted = strlen(value);
    const char *cursor = csv;
    while (*cursor) {
        while (*cursor == ',')
            cursor++;
        const char *end = strchr(cursor, ',');
        size_t length = end ? (size_t)(end - cursor) : strlen(cursor);
        if (length == wanted && strncmp(cursor, value, wanted) == 0)
            return true;
        if (!end)
            return false;
        cursor = end + 1;
    }
    return false;
}

static void decode_mount_field(const char *input, char *output, size_t size) {
    size_t used = 0;
    while (*input && used + 1 < size) {
        if (input[0] == '\\' && isdigit((unsigned char)input[1]) &&
            isdigit((unsigned char)input[2]) &&
            isdigit((unsigned char)input[3])) {
            unsigned int value = (unsigned int)(input[1] - '0') * 64U +
                                 (unsigned int)(input[2] - '0') * 8U +
                                 (unsigned int)(input[3] - '0');
            output[used++] = (char)value;
            input += 4;
        } else {
            output[used++] = *input++;
        }
    }
    output[used] = '\0';
}

static bool find_membership(int version, char *output, size_t size) {
    char *buffer = malloc(READ_BUFFER_SIZE);
    if (!buffer)
        return false;
    if (raw_read_file("/proc/self/cgroup", buffer, READ_BUFFER_SIZE) < 0) {
        free(buffer);
        return false;
    }

    bool found = false;
    char *save_line = NULL;
    for (char *line = strtok_r(buffer, "\n", &save_line); line;
         line = strtok_r(NULL, "\n", &save_line)) {
        char *first = strchr(line, ':');
        if (!first)
            continue;
        char *second = strchr(first + 1, ':');
        if (!second)
            continue;
        *first = '\0';
        *second = '\0';
        const char *controllers = first + 1;
        const char *path = second + 1;
        if ((version == 2 && controllers[0] == '\0') ||
            (version == 1 && csv_contains(controllers, "memory"))) {
            int count = snprintf(output, size, "/%s", path[0] == '/' ? path + 1 : path);
            found = count >= 0 && (size_t)count < size;
            break;
        }
    }
    free(buffer);
    return found;
}

static bool find_mount(int version, char *root, size_t root_size, char *point,
                       size_t point_size) {
    char *buffer = malloc(READ_BUFFER_SIZE);
    if (!buffer)
        return false;
    if (raw_read_file("/proc/self/mountinfo", buffer, READ_BUFFER_SIZE) < 0) {
        free(buffer);
        return false;
    }

    bool found = false;
    char *save_line = NULL;
    for (char *line = strtok_r(buffer, "\n", &save_line); line;
         line = strtok_r(NULL, "\n", &save_line)) {
        char *separator = strstr(line, " - ");
        if (!separator)
            continue;
        *separator = '\0';
        char *right = separator + 3;

        char *left_fields[6] = {0};
        size_t left_count = 0;
        char *save_left = NULL;
        for (char *token = strtok_r(line, " ", &save_left);
             token && left_count < 6;
             token = strtok_r(NULL, " ", &save_left))
            left_fields[left_count++] = token;
        if (left_count < 6)
            continue;

        char *save_right = NULL;
        char *fs_type = strtok_r(right, " ", &save_right);
        (void)strtok_r(NULL, " ", &save_right); /* mount source */
        char *super_options = strtok_r(NULL, " ", &save_right);
        if (!fs_type || !super_options)
            continue;
        bool matches = version == 2 ? strcmp(fs_type, "cgroup2") == 0
                                    : strcmp(fs_type, "cgroup") == 0 &&
                                          csv_contains(super_options, "memory");
        if (!matches)
            continue;
        decode_mount_field(left_fields[3], root, root_size);
        decode_mount_field(left_fields[4], point, point_size);
        found = true;
        break;
    }
    free(buffer);
    return found;
}

static bool path_has_prefix(const char *path, const char *prefix) {
    size_t length = strlen(prefix);
    if (length == 0)
        return false;
    if (strncmp(path, prefix, length) != 0)
        return false;
    return path[length] == '\0' || prefix[length - 1] == '/' || path[length] == '/';
}

static bool map_membership(const char *membership, const char *mount_root,
                           const char *mount_point, char *output, size_t size) {
    const char *relative = membership;
    if (strcmp(membership, "/") == 0) {
        relative = "";
    } else if (strcmp(mount_root, "/") == 0) {
        relative = membership + 1;
    } else if (strcmp(membership, mount_root) == 0) {
        relative = "";
    } else if (path_has_prefix(membership, mount_root)) {
        relative = membership + strlen(mount_root);
        while (*relative == '/')
            relative++;
    } else {
        /* Namespace-relative membership below a hidden mount root. */
        relative = membership[0] == '/' ? membership + 1 : membership;
    }
    return join_path(output, size, mount_point, relative);
}

static bool find_memory_location(struct cgroup_location *location) {
    for (int version = 2; version >= 1; version--) {
        struct cgroup_location candidate = {.version = version};
        if (!find_membership(version, candidate.membership,
                             sizeof(candidate.membership)))
            continue;
        if (!find_mount(version, candidate.mount_root,
                        sizeof(candidate.mount_root), candidate.mount_point,
                        sizeof(candidate.mount_point)))
            continue;
        if (!map_membership(candidate.membership, candidate.mount_root,
                            candidate.mount_point, candidate.current,
                            sizeof(candidate.current)))
            continue;
        char marker[PATH_MAX];
        const char *name = version == 2 ? "memory.current" : "memory.usage_in_bytes";
        if (!join_path(marker, sizeof(marker), candidate.current, name) ||
            !raw_file_exists(marker))
            continue;
        *location = candidate;
        return true;
    }
    return false;
}

static bool parse_unsigned(const char *text, unsigned long long *value,
                           bool limit_value) {
    while (isspace((unsigned char)*text))
        text++;
    if (!*text || strcmp(text, "max") == 0 || strcmp(text, "max\n") == 0 ||
        strcmp(text, "-1") == 0 || strcmp(text, "-1\n") == 0)
        return false;
    if (*text == '-')
        return false;
    errno = 0;
    char *end = NULL;
    unsigned long long parsed = strtoull(text, &end, 10);
    if (errno || end == text)
        return false;
    while (*end && isspace((unsigned char)*end))
        end++;
    if (*end)
        return false;
    if (limit_value && parsed >= UNLIMITED_THRESHOLD)
        return false;
    *value = parsed;
    return true;
}

static bool read_number(const char *directory, const char *filename,
                        unsigned long long *value, bool limit_value) {
    char path[PATH_MAX];
    char buffer[128];
    if (!join_path(path, sizeof(path), directory, filename) ||
        raw_read_file(path, buffer, sizeof(buffer)) < 0)
        return false;
    return parse_unsigned(buffer, value, limit_value);
}

static bool read_stat(const char *directory, const char *key,
                      unsigned long long *value) {
    char path[PATH_MAX];
    if (!join_path(path, sizeof(path), directory, "memory.stat"))
        return false;
    char *buffer = malloc(READ_BUFFER_SIZE);
    if (!buffer)
        return false;
    if (raw_read_file(path, buffer, READ_BUFFER_SIZE) < 0) {
        free(buffer);
        return false;
    }

    bool found = false;
    size_t key_length = strlen(key);
    char *save = NULL;
    for (char *line = strtok_r(buffer, "\n", &save); line;
         line = strtok_r(NULL, "\n", &save)) {
        if (strncmp(line, key, key_length) != 0 ||
            !isspace((unsigned char)line[key_length]))
            continue;
        found = parse_unsigned(line + key_length, value, false);
        break;
    }
    free(buffer);
    return found;
}

static bool parent_directory(char *path, const char *mount_point) {
    if (strcmp(path, mount_point) == 0)
        return false;
    size_t mount_length = strlen(mount_point);
    if (!path_has_prefix(path, mount_point))
        return false;
    size_t length = strlen(path);
    while (length > mount_length && path[length - 1] == '/')
        path[--length] = '\0';
    char *slash = strrchr(path, '/');
    if (!slash || (size_t)(slash - path) < mount_length)
        return false;
    *slash = '\0';
    return true;
}

static bool effective_limit(const struct cgroup_location *location,
                            const char *filename, unsigned long long host_limit,
                            unsigned long long *result, char *source,
                            size_t source_size) {
    bool have_value = host_limit > 0;
    bool have_source = false;
    unsigned long long effective = host_limit;
    char directory[PATH_MAX];
    if (snprintf(directory, sizeof(directory), "%s", location->current) < 0)
        return false;

    while (true) {
        unsigned long long candidate;
        if (read_number(directory, filename, &candidate, true) &&
            (!have_value || candidate <= effective)) {
            effective = candidate;
            have_value = true;
            have_source = true;
            snprintf(source, source_size, "%s", directory);
        }
        if (strcmp(directory, location->mount_point) == 0 ||
            !parent_directory(directory, location->mount_point))
            break;
    }
    if (!have_value)
        return false;
    *result = effective;
    return have_source;
}

static bool meminfo_value(const char *meminfo, const char *key,
                          unsigned long long *bytes) {
    size_t key_length = strlen(key);
    const char *line = meminfo;
    while (line && *line) {
        if (strncmp(line, key, key_length) == 0 && line[key_length] == ':') {
            unsigned long long kib;
            if (!parse_unsigned(line + key_length + 1, &kib, false)) {
                /* parse_unsigned rejects the trailing kB, so parse directly. */
                char *end = NULL;
                errno = 0;
                kib = strtoull(line + key_length + 1, &end, 10);
                if (errno || end == line + key_length + 1)
                    return false;
            }
            if (kib > ULLONG_MAX / 1024ULL)
                return false;
            *bytes = kib * 1024ULL;
            return true;
        }
        line = strchr(line, '\n');
        if (line)
            line++;
    }
    return false;
}

static bool collect_snapshot(struct memory_snapshot *snapshot) {
    char *host_meminfo = malloc(READ_BUFFER_SIZE);
    if (!host_meminfo)
        return false;
    if (raw_read_file("/proc/meminfo", host_meminfo, READ_BUFFER_SIZE) < 0) {
        free(host_meminfo);
        return false;
    }

    unsigned long long host_total = 0;
    unsigned long long host_swap_total = 0;
    unsigned long long host_swap_free = 0;
    bool host_ok = meminfo_value(host_meminfo, "MemTotal", &host_total);
    (void)meminfo_value(host_meminfo, "SwapTotal", &host_swap_total);
    (void)meminfo_value(host_meminfo, "SwapFree", &host_swap_free);
    free(host_meminfo);
    if (!host_ok || !host_total)
        return false;

    struct cgroup_location location;
    if (!find_memory_location(&location))
        return false;

    const char *limit_name = location.version == 2 ? "memory.max"
                                                    : "memory.limit_in_bytes";
    const char *usage_name = location.version == 2 ? "memory.current"
                                                    : "memory.usage_in_bytes";
    unsigned long long limit = host_total;
    char limit_source[PATH_MAX] = {0};
    bool cgroup_source = effective_limit(&location, limit_name, host_total,
                                         &limit, limit_source,
                                         sizeof(limit_source));

    if (location.version == 1) {
        unsigned long long hierarchical;
        if (read_stat(location.current, "hierarchical_memory_limit",
                      &hierarchical) &&
            hierarchical > 0 && hierarchical < UNLIMITED_THRESHOLD &&
            hierarchical <= limit) {
            limit = hierarchical;
            snprintf(limit_source, sizeof(limit_source), "%s", location.current);
            cgroup_source = true;
        }
    }
    if (!cgroup_source)
        return false;

    unsigned long long current;
    if (!read_number(limit_source, usage_name, &current, false))
        return false;
    unsigned long long inactive = 0;
    const char *inactive_key =
        location.version == 2 ? "inactive_file" : "total_inactive_file";
    if (!read_stat(limit_source, inactive_key, &inactive) && location.version == 1)
        (void)read_stat(limit_source, "inactive_file", &inactive);
    if (inactive > current)
        inactive = current;

    unsigned long long swap_limit = host_swap_total;
    unsigned long long swap_current =
        host_swap_total > host_swap_free ? host_swap_total - host_swap_free : 0;
    if (location.version == 2 && host_swap_total > 0) {
        char swap_source[PATH_MAX] = {0};
        unsigned long long candidate = host_swap_total;
        if (effective_limit(&location, "memory.swap.max", host_swap_total,
                            &candidate, swap_source, sizeof(swap_source)) &&
            candidate < host_swap_total) {
            swap_limit = candidate;
            if (!read_number(swap_source, "memory.swap.current", &swap_current,
                             false))
                swap_current = 0;
        }
    } else if (location.version == 1 && host_swap_total > 0) {
        char memsw_source[PATH_MAX] = {0};
        unsigned long long memsw_limit = 0;
        if (effective_limit(&location, "memory.memsw.limit_in_bytes", 0,
                            &memsw_limit, memsw_source,
                            sizeof(memsw_source)) &&
            memsw_limit >= limit) {
            unsigned long long allowed_swap = memsw_limit - limit;
            swap_limit = allowed_swap < host_swap_total ? allowed_swap
                                                        : host_swap_total;
            unsigned long long memsw_current = 0;
            unsigned long long memory_current = 0;
            if (read_number(memsw_source, "memory.memsw.usage_in_bytes",
                            &memsw_current, false) &&
                read_number(memsw_source, "memory.usage_in_bytes",
                            &memory_current, false) &&
                memsw_current > memory_current)
                swap_current = memsw_current - memory_current;
            else
                swap_current = 0;
        }
    }
    if (swap_current > swap_limit)
        swap_current = swap_limit;

    *snapshot = (struct memory_snapshot){
        .host_total = host_total,
        .limit = limit,
        .current = current,
        .inactive_file = inactive,
        .swap_limit = swap_limit,
        .swap_current = swap_current,
    };
    return true;
}

static bool append_bytes(char *output, size_t capacity, size_t *used,
                         const char *data, size_t length) {
    if (*used + length + 1 > capacity)
        return false;
    memcpy(output + *used, data, length);
    *used += length;
    output[*used] = '\0';
    return true;
}

static bool append_kib_line(char *output, size_t capacity, size_t *used,
                            const char *key, unsigned long long bytes) {
    char line[128];
    int count = snprintf(line, sizeof(line), "%-18s %llu kB\n", key,
                         bytes / 1024ULL);
    return count > 0 && (size_t)count < sizeof(line) &&
           append_bytes(output, capacity, used, line, (size_t)count);
}

static const char *line_key(const char *line, size_t length,
                            const char *const *keys, size_t key_count,
                            size_t *index) {
    for (size_t i = 0; i < key_count; i++) {
        size_t key_length = strlen(keys[i]);
        if (length >= key_length && strncmp(line, keys[i], key_length) == 0) {
            *index = i;
            return keys[i];
        }
    }
    return NULL;
}

static char *synthetic_meminfo(size_t *output_length) {
    struct memory_snapshot snapshot;
    if (!collect_snapshot(&snapshot))
        return NULL;

    char *original = malloc(READ_BUFFER_SIZE);
    if (!original)
        return NULL;
    ssize_t original_length =
        raw_read_file("/proc/meminfo", original, READ_BUFFER_SIZE);
    if (original_length < 0) {
        free(original);
        return NULL;
    }

    size_t capacity = (size_t)original_length + 4096U;
    char *output = calloc(1, capacity);
    if (!output) {
        free(original);
        return NULL;
    }

    unsigned long long free_bytes =
        snapshot.limit > snapshot.current ? snapshot.limit - snapshot.current : 0;
    unsigned long long working = snapshot.current - snapshot.inactive_file;
    unsigned long long available =
        snapshot.limit > working ? snapshot.limit - working : 0;
    unsigned long long swap_free = snapshot.swap_limit > snapshot.swap_current
                                       ? snapshot.swap_limit - snapshot.swap_current
                                       : 0;

    const char *const keys[] = {
        "MemTotal:",   "MemFree:",    "MemAvailable:", "Buffers:",
        "Cached:",     "SReclaimable:", "Shmem:",      "SwapTotal:",
        "SwapFree:",   "SwapCached:",
    };
    const unsigned long long values[] = {
        snapshot.limit, free_bytes, available, 0, snapshot.inactive_file,
        0,              0,          snapshot.swap_limit, swap_free, 0,
    };

    size_t used = 0;
    const char *cursor = original;
    const char *end = original + original_length;
    while (cursor < end) {
        const char *newline = memchr(cursor, '\n', (size_t)(end - cursor));
        size_t length = newline ? (size_t)(newline - cursor + 1)
                                : (size_t)(end - cursor);
        size_t key_index = 0;
        const char *key = line_key(cursor, length, keys,
                                   sizeof(keys) / sizeof(keys[0]), &key_index);
        bool ok = key ? append_kib_line(output, capacity, &used, key,
                                        values[key_index])
                      : append_bytes(output, capacity, &used, cursor, length);
        if (!ok) {
            free(original);
            free(output);
            return NULL;
        }
        cursor += length;
    }
    free(original);
    *output_length = used;
    return output;
}

static int create_meminfo_fd(void) {
    size_t length = 0;
    char *content = synthetic_meminfo(&length);
    if (!content)
        return -1;
#ifdef SYS_memfd_create
    int fd = (int)syscall(SYS_memfd_create, "pod-meminfo", MFD_CLOEXEC);
#else
    errno = ENOSYS;
    int fd = -1;
#endif
    if (fd < 0) {
        free(content);
        return -1;
    }
    size_t written = 0;
    while (written < length) {
        ssize_t count = write(fd, content + written, length - written);
        if (count > 0) {
            written += (size_t)count;
            continue;
        }
        if (count < 0 && errno == EINTR)
            continue;
        close(fd);
        free(content);
        return -1;
    }
    free(content);
    if (lseek(fd, 0, SEEK_SET) < 0) {
        close(fd);
        return -1;
    }
    return fd;
}

static bool readable_open_flags(int flags) {
    if ((flags & O_ACCMODE) != O_RDONLY)
        return false;
#ifdef O_PATH
    if (flags & O_PATH)
        return false;
#endif
    return (flags & O_DIRECTORY) == 0;
}

static bool flags_have_mode(int flags) {
    if (flags & O_CREAT)
        return true;
#ifdef O_TMPFILE
    if ((flags & O_TMPFILE) == O_TMPFILE)
        return true;
#endif
    return false;
}

FILE *fopen(const char *path, const char *mode) {
    resolve_symbols();
    if (!real_fopen)
        return NULL;
    if (hook_active || disabled() || !is_meminfo_path(path) || !mode ||
        mode[0] != 'r' || strchr(mode, '+'))
        return real_fopen(path, mode);

    hook_active++;
    int fd = create_meminfo_fd();
    hook_active--;
    if (fd < 0)
        return real_fopen(path, mode);
    FILE *stream = fdopen(fd, mode);
    if (!stream)
        close(fd);
    return stream;
}

FILE *fopen64(const char *path, const char *mode) {
    resolve_symbols();
    fopen_function fallback = real_fopen64 ? real_fopen64 : real_fopen;
    if (!fallback)
        return NULL;
    if (hook_active || disabled() || !is_meminfo_path(path) || !mode ||
        mode[0] != 'r' || strchr(mode, '+'))
        return fallback(path, mode);

    hook_active++;
    int fd = create_meminfo_fd();
    hook_active--;
    if (fd < 0)
        return fallback(path, mode);
    FILE *stream = fdopen(fd, mode);
    if (!stream)
        close(fd);
    return stream;
}

static int hooked_open(const char *path, int flags, mode_t mode, bool have_mode,
                       open_function fallback) {
    if (!fallback) {
        errno = ENOSYS;
        return -1;
    }
    if (!hook_active && !disabled() && is_meminfo_path(path) &&
        readable_open_flags(flags)) {
        hook_active++;
        int fd = create_meminfo_fd();
        hook_active--;
        if (fd >= 0)
            return fd;
    }
    return have_mode ? fallback(path, flags, mode) : fallback(path, flags);
}

int open(const char *path, int flags, ...) {
    resolve_symbols();
    mode_t mode = 0;
    bool have_mode = flags_have_mode(flags);
    if (have_mode) {
        va_list arguments;
        va_start(arguments, flags);
        mode = (mode_t)va_arg(arguments, int);
        va_end(arguments);
    }
    return hooked_open(path, flags, mode, have_mode, real_open);
}

int open64(const char *path, int flags, ...) {
    resolve_symbols();
    mode_t mode = 0;
    bool have_mode = flags_have_mode(flags);
    if (have_mode) {
        va_list arguments;
        va_start(arguments, flags);
        mode = (mode_t)va_arg(arguments, int);
        va_end(arguments);
    }
    return hooked_open(path, flags, mode, have_mode,
                       real_open64 ? real_open64 : real_open);
}

static int hooked_openat(int directory_fd, const char *path, int flags,
                         mode_t mode, bool have_mode,
                         openat_function fallback) {
    if (!fallback) {
        errno = ENOSYS;
        return -1;
    }
    if (!hook_active && !disabled() && is_meminfo_path(path) &&
        readable_open_flags(flags)) {
        hook_active++;
        int fd = create_meminfo_fd();
        hook_active--;
        if (fd >= 0)
            return fd;
    }
    return have_mode ? fallback(directory_fd, path, flags, mode)
                     : fallback(directory_fd, path, flags);
}

int openat(int directory_fd, const char *path, int flags, ...) {
    resolve_symbols();
    mode_t mode = 0;
    bool have_mode = flags_have_mode(flags);
    if (have_mode) {
        va_list arguments;
        va_start(arguments, flags);
        mode = (mode_t)va_arg(arguments, int);
        va_end(arguments);
    }
    return hooked_openat(directory_fd, path, flags, mode, have_mode, real_openat);
}

int openat64(int directory_fd, const char *path, int flags, ...) {
    resolve_symbols();
    mode_t mode = 0;
    bool have_mode = flags_have_mode(flags);
    if (have_mode) {
        va_list arguments;
        va_start(arguments, flags);
        mode = (mode_t)va_arg(arguments, int);
        va_end(arguments);
    }
    return hooked_openat(directory_fd, path, flags, mode, have_mode,
                         real_openat64 ? real_openat64 : real_openat);
}

/* glibc fortify entry points used when the caller supplies no creation mode.
 * Delegating non-target paths to the real fortify functions preserves their
 * diagnostics for invalid O_CREAT/O_TMPFILE calls and avoids reading a
 * nonexistent vararg in our ordinary open hooks. */
int __open_2(const char *path, int flags) {
    resolve_symbols();
    if (!hook_active && !disabled() && is_meminfo_path(path) &&
        readable_open_flags(flags))
        return hooked_open(path, flags, 0, false, real_open);
    if (real___open_2)
        return real___open_2(path, flags);
    if (!flags_have_mode(flags) && real_open)
        return real_open(path, flags);
    errno = EINVAL;
    return -1;
}

int __open64_2(const char *path, int flags) {
    resolve_symbols();
    if (!hook_active && !disabled() && is_meminfo_path(path) &&
        readable_open_flags(flags))
        return hooked_open(path, flags, 0, false,
                           real_open64 ? real_open64 : real_open);
    if (real___open64_2)
        return real___open64_2(path, flags);
    if (!flags_have_mode(flags)) {
        open_function fallback = real_open64 ? real_open64 : real_open;
        if (fallback)
            return fallback(path, flags);
    }
    errno = EINVAL;
    return -1;
}

int __openat_2(int directory_fd, const char *path, int flags) {
    resolve_symbols();
    if (!hook_active && !disabled() && is_meminfo_path(path) &&
        readable_open_flags(flags))
        return hooked_openat(directory_fd, path, flags, 0, false, real_openat);
    if (real___openat_2)
        return real___openat_2(directory_fd, path, flags);
    if (!flags_have_mode(flags) && real_openat)
        return real_openat(directory_fd, path, flags);
    errno = EINVAL;
    return -1;
}

int __openat64_2(int directory_fd, const char *path, int flags) {
    resolve_symbols();
    if (!hook_active && !disabled() && is_meminfo_path(path) &&
        readable_open_flags(flags))
        return hooked_openat(directory_fd, path, flags, 0, false,
                             real_openat64 ? real_openat64 : real_openat);
    if (real___openat64_2)
        return real___openat64_2(directory_fd, path, flags);
    if (!flags_have_mode(flags)) {
        openat_function fallback = real_openat64 ? real_openat64 : real_openat;
        if (fallback)
            return fallback(directory_fd, path, flags);
    }
    errno = EINVAL;
    return -1;
}
